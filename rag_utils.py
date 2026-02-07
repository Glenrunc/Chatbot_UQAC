"""
Module RAG réutilisable pour le Chatbot UQAC.
Contient les classes et fonctions pour la récupération et le reranking de documents.
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field, ConfigDict
import json
import numpy as np
from typing import List
from pathlib import Path


class JSONLRetriever(BaseRetriever):
    """Retriever personnalisé basé sur un corpus JSONL avec embeddings pré-calculés."""
    corpus: List[dict] = Field(default_factory=list)
    embeddings_model: OllamaEmbeddings = Field(default=None)
    k: int = Field(default=40)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calcule la similarité cosinus entre deux vecteurs."""
        a, b = np.array(vec1), np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Récupère les documents les plus pertinents pour une requête."""
        query_embedding = self.embeddings_model.embed_query(query)
        
        # Calculer les similarités pour tous les items
        similarities = [
            (self._cosine_similarity(query_embedding, item["embedding"]), item)
            for item in self.corpus
            if "embedding" in item
        ]
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Créer les documents
        return [
            Document(
                page_content=item.get("chunk", ""),
                metadata={
                    "source": item.get("url", ""),
                    "title": item.get("title", ""),
                    "section": item.get("section", ""),
                },
            )
            for _, item in similarities[:self.k]
        ]


def load_corpus(corpus_path: Path) -> List[dict]:
    """Charge le corpus depuis un fichier JSONL."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_rag_chain(
    corpus_path: Path,
    model_llm: str = "qwen3:8b",
    model_embedding: str = "bge-m3",
    model_reranker: str = "ms-marco-MultiBERT-L-12",
    retriever_k: int = 40,
    retriever_top_n: int = 8,
):
    """Construit la chaîne RAG complète."""
    
    # Chargement du corpus
    corpus_data = load_corpus(corpus_path)

    # Modèle d'embeddings
    embeddings = OllamaEmbeddings(model=model_embedding)

    # Retriever de base
    base_retriever = JSONLRetriever(
        corpus=corpus_data, 
        embeddings_model=embeddings, 
        k=retriever_k
    )

    # Reranker pour améliorer la pertinence
    compressor = FlashrankRerank(model=model_reranker, top_n=retriever_top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )

    # Prompts
    history_system_prompt = (
        "Compte tenu de l'historique de la conversation et de la dernière question de l'utilisateur "
        "(qui peut faire référence au contexte passé), formule une question autonome "
        "qui peut être comprise sans l'historique de la conversation. "
        "Ne réponds PAS à la question, reformule-la simplement si nécessaire en FRANÇAIS, "
        "sinon renvoie-la telle quelle."
    )
    history_prompt = ChatPromptTemplate.from_messages(
        [("system", history_system_prompt), ("placeholder", "{chat_history}"), ("human", "{input}")]
    )

    qa_system_prompt = (
        "Tu es un assistant expert de l'UQAC chargé de guider les utilisateurs. "
        "Utilise les documents suivants pour répondre. Si tu ne sais pas, dis-le.\n\n"
        "Instructions strictes :\n"
        "1. STRUCTURE OBLIGATOIRE : Sépare ta réponse en deux parties distinctes :\n"
        "   - [THINKING] : Ta réflexion interne, analyse des documents, raisonnement (2-3 phrases max)\n"
        "   - [ANSWER] : Ta réponse finale concise et claire pour l'utilisateur\n\n"
        "2. Dans [ANSWER], donne une réponse concise en français avec un court résumé.\n"
        "3. N'inclus PAS de liens ou sources dans ta réponse (ils seront affichés séparément).\n\n"
        "Contexte :\n{context}\n\n"
        "Format obligatoire :\n"
        "[THINKING]\n<ton analyse des documents et raisonnement>\n\n"
        "[ANSWER]\n<ta réponse finale claire et concise>"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt), ("placeholder", "{chat_history}"), ("human", "{input}")]
    )

    # LLM
    llm = ChatOllama(
        model=model_llm, 
        temperature=0, 
        base_url="http://localhost:11434"
    )

    # Construction des chaînes
    history_aware_retriever = create_history_aware_retriever(
        llm, compression_retriever, history_prompt
    )
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Source: {source}\nContenu: {page_content}",
    )
    question_answer_chain = create_stuff_documents_chain(
        llm, qa_prompt, document_prompt=document_prompt
    )
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain
