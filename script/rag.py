from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
embeddings = OllamaEmbeddings(model="bge-m3")
vectorstore = Chroma(persist_directory="../data/chroma_db", embedding_function=embeddings)

# retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 40})

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
    model="llama3",
    temperature=0,
    base_url="http://localhost:11434"
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

question1 = "Quel est la responsabilité du doyen des études ?"
print(f"MOI: {question1}")

reponse1 = conversational_rag_chain.invoke(
    {"input": question1},
    config={"configurable": {"session_id": session_id}}
)
print(f"AI: {reponse1['answer']}\n")

question2 = "Et qui est son supérieur ?" # check si la memoire fonctionne 
print(f"MOI: {question2}")

reponse2 = conversational_rag_chain.invoke(
    {"input": question2},
    config={"configurable": {"session_id": session_id}}
)
print(f"AI: {reponse2['answer']}")
