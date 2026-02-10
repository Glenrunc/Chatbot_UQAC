"""
Script de test pour le système RAG du Chatbot UQAC.
Utilise le module rag_utils pour éviter la duplication de code.
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer rag_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from rag_utils import build_rag_chain
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

# Chemin vers le corpus
corpus_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "corpus.jsonl"

# Construire la chaîne RAG
rag_chain = build_rag_chain(
    corpus_path=corpus_path,
    model_llm="qwen3:8b",
    model_embedding="bge-m3",
    model_reranker="ms-marco-MultiBERT-L-12",
    retriever_k=40,
    retriever_top_n=8,
)

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