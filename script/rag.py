from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field, ConfigDict
import json
import numpy as np
from typing import List
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)


# Charger le corpus JSONL
def load_corpus(path: str):
    corpus = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus

corpus_path = "../data/processed/corpus.jsonl"
corpus_data = load_corpus(corpus_path)

# Embeddings model
embeddings = OllamaEmbeddings(model="bge-m3")

# Retriever personnalisé basé sur le corpus JSONL
class JSONLRetriever(BaseRetriever):
    corpus: List[dict] = Field(default_factory=list)
    embeddings_model: OllamaEmbeddings = Field(default=None)
    k: int = Field(default=40)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        a = np.array(vec1)
        b = np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Obtenir l'embedding de la requête
        query_embedding = self.embeddings_model.embed_query(query)
        
        # Calculer les similarités
        similarities = []
        for item in self.corpus:
            if "embedding" in item:
                sim = self._cosine_similarity(query_embedding, item["embedding"])
                similarities.append((sim, item))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Retourner les k documents les plus pertinents
        documents = []
        for sim, item in similarities[:self.k]:
            doc = Document(
                page_content=item.get("chunk", ""),
                metadata={
                    "source": item.get("url", ""),
                    "title": item.get("title", ""),
                    "section": item.get("section", ""),
                }
            )
            documents.append(doc)
        
        return documents

# Créer le retriever
base_retriever = JSONLRetriever(
    corpus=corpus_data,
    embeddings_model=embeddings,
    k=40
)

# reranker
compressor = FlashrankRerank(
    model="ms-marco-MultiBERT-L-12",
    top_n=8
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)


# la mémoire
history_system_prompt = """Compte tenu de l'historique de la conversation et de la dernière question de l'utilisateur 
(qui peut faire référence au contexte passé), formule une question autonome 
qui peut être comprise sans l'historique de la conversation. 
Ne réponds PAS à la question, reformule-la simplement si nécessaire en FRANÇAIS, 
sinon renvoie-la telle quelle."""

history_prompt = ChatPromptTemplate.from_messages([
    ("system", history_system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ]
)

qa_system_prompt = """Tu es un assistant expert chargé de guider les utilisateurs. 
Utilise les documents suivants pour répondre. Si tu ne sais pas, dis-le.

Instructions strictes :
1. Fais un court résumé.
2. Source : Copie le lien EXACTEMENT tel qu'il apparaît dans "Source" dans le contexte. 
   Ne modifie rien (pas d'accents, pas de correction).
3. Réponse concise en français.

Contexte :
{context}

Format :
{{user: <question>
assistant: <réponse + lien>}}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)



llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    base_url="http://localhost:11434",
   
)

# CHAÎNE RAG
history_aware_retriever = create_history_aware_retriever(
    llm, 
    compression_retriever, 
    history_prompt
)

# Chaîne de réponse
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Source: {source}\nContenu: {page_content}"
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt=document_prompt)

# Chaîne globale (Recherche + Réponse)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Gestion de la session
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# La chaîne finale appelable
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# Simulation d'une conversation
session_id = "session_test_1"

def format_sources(response):
    """Extrait et formate les sources utilisées depuis les documents récupérés."""
    sources = []
    if "context" in response:
        for doc in response["context"]:
            source_url = doc.metadata.get("source", "")
            title = doc.metadata.get("title", "Sans titre")
            if source_url and source_url not in [s["url"] for s in sources]:
                sources.append({"url": source_url, "title": title})
    return sources

def print_response_with_sources(question, response):
    """Affiche la réponse avec les sources utilisées."""
    print(f"AI: {response['answer']}")
    
    sources = format_sources(response)
    if sources:
        print("\nSources utilisées (depuis la base de données locale) :")
        for i, src in enumerate(sources, 1):
            print(f"   {i}. {src['title']}")
            print(f"      {src['url']}")
    print()

question1 = "Quelle est la politique d'éthique de la recherche avec des êtres humains  ?"
print(f"MOI: {question1}")

reponse1 = conversational_rag_chain.invoke(
    {"input": question1},
    config={"configurable": {"session_id": session_id}}
)
print_response_with_sources(question1, reponse1)

# question2 = "Détails pour les admissions en art ?" # check si la memoire fonctionne 
# print(f"MOI: {question2}")

# reponse2 = conversational_rag_chain.invoke(
#     {"input": question2},
#     config={"configurable": {"session_id": session_id}}
# )
# print_response_with_sources(question2, reponse2)