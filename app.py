import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import List, Tuple
from pathlib import Path
import logging
import uuid
import json
from datetime import datetime

# Import du module RAG
from rag_utils import build_rag_chain

logging.getLogger("httpx").setLevel(logging.WARNING)

# Constants
MODEL_LLM = "qwen3:8b"
MODEL_EMBEDDING = "bge-m3"
MODEL_RERANKER = "ms-marco-MultiBERT-L-12"
RETRIEVER_K = 40
RETRIEVER_TOP_N = 8
MAX_TITLE_LENGTH = 30
THINKING_PREVIEW_LENGTH = 60
MAX_CONVERSATIONS = 10
CONVERSATIONS_FILE = Path(__file__).resolve().parent / "data" / "conversations.json"

# Page config
st.set_page_config(
    page_title="Chatbot UQAC",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

def load_css():
    """Charge le fichier CSS externe."""
    css_path = Path(__file__).resolve().parent / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

load_css()


# RAG pipeline
@st.cache_resource(show_spinner="Chargement du corpus et du modèle …")
def get_rag_chain():
    """Construit la chaîne RAG (mise en cache)."""
    corpus_path = Path(__file__).resolve().parent / "data" / "processed" / "corpus.jsonl"
    return build_rag_chain(
        corpus_path=corpus_path,
        model_llm=MODEL_LLM,
        model_embedding=MODEL_EMBEDDING,
        model_reranker=MODEL_RERANKER,
        retriever_k=RETRIEVER_K,
        retriever_top_n=RETRIEVER_TOP_N,
    )


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Récupère ou crée l'historique de chat pour une session."""
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = ChatMessageHistory()
    return st.session_state.chat_histories[session_id]


def format_sources(response) -> List[dict]:
    """Extrait les sources uniques depuis la réponse RAG."""
    sources = []
    seen = set()
    if "context" in response:
        for doc in response["context"]:
            url = doc.metadata.get("source", "")
            title = doc.metadata.get("title", "Sans titre")
            if url and url not in seen:
                seen.add(url)
                sources.append({"url": url, "title": title})
    return sources


def truncate_title(title: str, max_length: int = MAX_TITLE_LENGTH) -> str:
    """Tronque un titre avec '...' s'il est trop long."""
    if len(title) > max_length:
        return title[:max_length] + "..."
    return title


def save_conversations():
    """Sauvegarde les conversations dans un fichier JSON."""
    try:
        # Convertir les datetime en string pour la sérialisation JSON
        conversations_serializable = {}
        for conv_id, conv_data in st.session_state.conversations.items():
            conversations_serializable[conv_id] = {
                "title": conv_data["title"],
                "date": conv_data["date"].isoformat(),
                "messages": conv_data["messages"]
            }
        
        # Créer le dossier data si absent
        CONVERSATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder
        with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations_serializable, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde des conversations: {e}")


def load_conversations() -> dict:
    """Charge les conversations depuis le fichier JSON."""
    try:
        if CONVERSATIONS_FILE.exists():
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                conversations_data = json.load(f)
            
            # Convertir les dates ISO string en datetime
            conversations = {}
            for conv_id, conv_data in conversations_data.items():
                conversations[conv_id] = {
                    "title": conv_data["title"],
                    "date": datetime.fromisoformat(conv_data["date"]),
                    "messages": conv_data["messages"]
                }
            return conversations
    except Exception as e:
        logging.error(f"Erreur lors du chargement des conversations: {e}")
    
    return {}


def remove_oldest_conversation():
    """Supprime la conversation la plus ancienne si on dépasse la limite."""
    if len(st.session_state.conversations) >= MAX_CONVERSATIONS:
        # Trouver la conversation la plus ancienne
        oldest_id = min(
            st.session_state.conversations.items(),
            key=lambda x: x[1]["date"]
        )[0]
        
        # Supprimer la conversation
        del st.session_state.conversations[oldest_id]
        if oldest_id in st.session_state.chat_histories:
            del st.session_state.chat_histories[oldest_id]


def parse_thinking_answer(text: str) -> Tuple[str, str]:
    """Sépare la partie THINKING de la partie ANSWER."""
    # Nettoyer les balises <think>
    text = text.replace("<think>", "").replace("</think>", "")
    
    thinking = ""
    answer = text
    
    # Cas 1 : Marqueurs [THINKING] et [ANSWER] présents
    if "[ANSWER]" in text:
        parts = text.split("[ANSWER]", 1)
        thinking = parts[0].replace("[THINKING]", "").strip()
        answer = parts[1].strip()
    
    # Cas 2 : Seulement [THINKING]
    elif "[THINKING]" in text:
        remaining = text.split("[THINKING]", 1)[1].strip()
        if "\n\n" in remaining:
            thinking, answer = remaining.split("\n\n", 1)
        else:
            thinking = remaining
            answer = ""
    
    # Cas 3 : Détection automatique si on détecte des patterns de thinking dans la réponse
    else:
        thinking_patterns = ["okay, the user", "let me start", "first, i need", "wait,", "i should"]
        if any(pattern in text.lower() for pattern in thinking_patterns):
            # Chercher la première ligne longue commençant par un article français
            lines = text.split("\n")
            for i, line in enumerate(lines):
                line = line.strip()
                if (len(line) > 50 and 
                    any(line.startswith(p) for p in ["Les ", "Le ", "La ", "L'", "À ", "Pour "]) and
                    not any(w in line.lower() for w in ["let me", "i need", "wait", "okay"])):
                    thinking = "\n".join(lines[:i]).strip()
                    answer = "\n".join(lines[i:]).strip()
                    break
    
    return thinking, answer


if "conversations" not in st.session_state:
    # Charger les conversations sauvegardées ou créer une nouvelle
    loaded_conversations = load_conversations()
    
    if loaded_conversations:
        st.session_state.conversations = loaded_conversations
    else:
        # Créer une première conversation vide
        st.session_state.conversations = {}
    
if "current_conversation_id" not in st.session_state:
    # Si des conversations existent, prendre la plus récente
    if st.session_state.conversations:
        most_recent = max(
            st.session_state.conversations.items(),
            key=lambda x: x[1]["date"]
        )
        st.session_state.current_conversation_id = most_recent[0]
    else:
        # Créer une première conversation
        new_id = str(uuid.uuid4())
        st.session_state.current_conversation_id = new_id
        st.session_state.conversations[new_id] = {
            "title": "",
            "date": datetime.now(),
            "messages": []
        }
        save_conversations()

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = st.session_state.current_conversation_id
else:
    st.session_state.session_id = st.session_state.current_conversation_id
    
st.session_state.messages = st.session_state.conversations[st.session_state.current_conversation_id]["messages"]

# Sidebar
with st.sidebar:
    st.markdown("### Chatbot UQAC")
    st.markdown("Assistant basé sur le **manuel de gestion** de l'UQAC.")
    st.divider()
    
    # Bouton nouvelle conversation
    if st.button("Nouvelle conversation", use_container_width=True):
        # Supprimer la plus ancienne si on atteint la limite
        remove_oldest_conversation()
        
        new_id = str(uuid.uuid4())
        st.session_state.conversations[new_id] = {
            "title": "",
            "date": datetime.now(),
            "messages": []
        }
        st.session_state.current_conversation_id = new_id
        st.session_state.session_id = new_id
        st.session_state.pending_prompt = None
        save_conversations()
        st.rerun()
    
    st.divider()
    
    # Liste des conversations
    st.markdown("**Historique**")
    
    # Trier les conversations par date (plus récentes en premier)
    sorted_convs = sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1]["date"],
        reverse=True
    )
    
    for conv_id, conv_data in sorted_convs:
        is_current = conv_id == st.session_state.current_conversation_id
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Bouton pour sélectionner la conversation
            display_title = conv_data['title'] if conv_data['title'] else "Nouvelle conversation"
            truncated_title = truncate_title(display_title, MAX_TITLE_LENGTH)
            button_label = f"{'▶ ' if is_current else ''}{truncated_title}"
            
            if st.button(button_label, key=f"conv_{conv_id}", use_container_width=True, 
                        type="primary" if is_current else "secondary"):
                if not is_current:
                    st.session_state.current_conversation_id = conv_id
                    st.session_state.session_id = conv_id
                    st.session_state.pending_prompt = None
                    st.rerun()
        
        with col2:
            # Bouton supprimer (sauf si c'est la seule conversation)
            if len(st.session_state.conversations) > 1:
                if st.button("🗑️", key=f"del_{conv_id}"):
                    del st.session_state.conversations[conv_id]
                    if conv_id in st.session_state.chat_histories:
                        del st.session_state.chat_histories[conv_id]
                    
                    # Si on supprime la conversation actuelle, passer à une autre
                    if conv_id == st.session_state.current_conversation_id:
                        st.session_state.current_conversation_id = list(st.session_state.conversations.keys())[0]
                        st.session_state.session_id = st.session_state.current_conversation_id
                    save_conversations()
                    st.rerun()
    
    st.divider()
    st.caption(f"Modèle : {MODEL_LLM} — Embeddings : {MODEL_EMBEDDING}")
    st.caption(f"Reranker : {MODEL_RERANKER}")


if not st.session_state.messages:
    st.markdown("# Chatbot UQAC")
    st.markdown(
        '<p class="subtitle">Posez vos questions sur le manuel de gestion de l\'UQAC</p>',
        unsafe_allow_html=True,
    )
    # Suggestions
    cols = st.columns([1, 1, 1])
    suggestions = [
        "Critères d'admission",
        "Régime des études",
        "Bourses disponibles",
    ]
    full_prompts = [
        "Quels sont les critères d'admission ?",
        "Comment fonctionne le régime des études ?",
        "Quelles sont les bourses disponibles ?",
    ]
    for i, (suggestion, full_prompt) in enumerate(zip(suggestions, full_prompts)):
        with cols[i]:
            if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_prompt = full_prompt
                st.rerun()

# Historique de la conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # Pour les messages assistant, afficher thinking et answer séparément
            thinking = msg.get("thinking", "")
            answer = msg.get("answer", msg["content"])
            
            if thinking:
                preview = truncate_title(thinking, THINKING_PREVIEW_LENGTH)
                with st.expander(f"💭 {preview}", expanded=False):
                    st.markdown(thinking)
            
            st.markdown(answer)
            
            if msg.get("sources"):
                with st.expander("📚 Sources consultées"):
                    for j, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**{j}. {src['title']}**  \n{src['url']}")

# Saisie de la question
prompt = None

# Vérifier s'il y a une question en attente depuis les suggestions
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif user_input := st.chat_input("Posez votre question …"):
    prompt = user_input

if prompt:
    # Mettre à jour le titre de la conversation si elle n'a pas encore de titre
    current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
    if not current_conv["title"] or current_conv["title"] == "":  # Pas encore de titre
        title = prompt[:50] + ("..." if len(prompt) > 50 else "")
        st.session_state.conversations[st.session_state.current_conversation_id]["title"] = title
    
    # Ajouter le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_conversations()
    
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    # Construire la chaîne RAG avec l'historique de la session
    rag_chain = get_rag_chain()
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Générer la réponse en streaming
    with st.chat_message("assistant", avatar="🎓"):
        # Containers pour l'affichage progressif
        thinking_container = st.empty()
        answer_container = st.empty()
        
        full_answer = ""
        thinking_text = ""
        answer_text = ""
        in_thinking = False
        in_answer = False
        
        # Stream la réponse
        for chunk in conversational_chain.stream(
            {"input": prompt},
            config={"configurable": {"session_id": st.session_state.session_id}},
        ):
            if "answer" in chunk:
                full_answer += chunk["answer"]
                
                # Détection des sections
                if "[THINKING]" in full_answer and not in_thinking:
                    in_thinking = True
                    
                if "[ANSWER]" in full_answer and not in_answer:
                    in_answer = True
                    # Extraire le thinking complet
                    parts = full_answer.split("[ANSWER]")
                    thinking_text = parts[0].replace("[THINKING]", "").replace("<think>", "").replace("</think>", "").strip()
                    answer_text = parts[1].replace("<think>", "").replace("</think>", "").strip() if len(parts) > 1 else ""
                    
                    # Afficher le thinking dans un expander fermé
                    if thinking_text:
                        preview = truncate_title(thinking_text, THINKING_PREVIEW_LENGTH)
                        with thinking_container:
                            with st.expander(f"💭 {preview}", expanded=False):
                                st.markdown(thinking_text)
                    thinking_container = st.empty()  # Clear le container
                    
                # Affichage progressif
                if in_answer:
                    # Extraire uniquement la partie answer
                    if "[ANSWER]" in full_answer:
                        answer_text = full_answer.split("[ANSWER]")[1].replace("<think>", "").replace("</think>", "").strip()
                    answer_container.markdown(answer_text)
                else:
                    # Avant d'atteindre [ANSWER], tout est considéré comme du thinking
                    # On affiche dans la bulle en italique
                    current_thinking = full_answer.replace("[THINKING]", "").replace("<think>", "").replace("</think>", "").strip()
                    if current_thinking:  # Ne rien afficher si vide
                        with thinking_container:
                            st.markdown(f'<div class="thinking-bubble">💭 <em>{current_thinking}</em></div>', unsafe_allow_html=True)
            
            # Stocker la réponse complète pour les sources
            if "context" in chunk:
                response_context = chunk["context"]
        
        # Parser la réponse finale
        thinking, answer = parse_thinking_answer(full_answer)
        
        # Affichage final
        if not in_answer and not in_thinking:
            if thinking:
                preview = truncate_title(thinking, THINKING_PREVIEW_LENGTH)
                with thinking_container:
                    with st.expander(f"💭 {preview}", expanded=False):
                        st.markdown(thinking)
            answer_container.markdown(answer)
        
        # Construire l'objet response pour les sources
        response = {"answer": full_answer, "context": response_context if 'response_context' in locals() else []}
        sources = format_sources(response)
        
        if sources:
            with st.expander("📚 Sources consultées"):
                for j, src in enumerate(sources, 1):
                    st.markdown(f"**{j}. {src['title']}**  \n{src['url']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_answer, "thinking": thinking, "answer": answer, "sources": sources}
    )
    
    # Sauvegarder les conversations
    save_conversations()
