# Chatbot_UQAC

Chatbot basé sur le manuel de gestion de l'UQAC utilisant RAG (Retrieval-Augmented Generation) avec Streamlit.

## Prérequis

- Python 3.12 ou supérieur
- [Ollama](https://ollama.com/download) installé et en cours d'exécution

## Installation

1. **Installer les dépendances Python** :
   ```bash
   pip install -r requirements.txt
   ```

2. **Installer et démarrer Ollama** :
   - Téléchargez et installez Ollama depuis [ollama.com/download](https://ollama.com/download)
   - Démarrez Ollama (il devrait se lancer automatiquement après installation)

3. **Télécharger les modèles requis** :
   ```bash
   ollama pull qwen3:8b
   ollama pull bge-m3
   ollama pull ms-marco-MultiBERT-L-12
   ```

## Démarrage du programme

1. **Assurez-vous qu'Ollama est en cours d'exécution** :
   - Vérifiez que le service Ollama est actif (une icône dans la barre des tâches ou via `ollama serve` en terminal)

2. **Lancer l'application Streamlit** :
   ```bash
   streamlit run app.py
   ```

3. **Accéder à l'application** :
   - Ouvrez votre navigateur à l'adresse affichée (généralement `http://localhost:8501`)

## Structure du projet

- `app.py` : Application principale Streamlit
- `rag_utils.py` : Utilitaires pour la chaîne RAG
- `data/` : Données du corpus et conversations
- `script/` : Scripts pour construire le corpus
- `requirements.txt` : Dépendances Python
- `style.css` : Styles CSS pour l'interface

## Dépannage

- **Erreur de connexion à Ollama** : Assurez-vous qu'Ollama est installé, démarré et accessible. Vérifiez avec `ollama list` que les modèles sont téléchargés.
- **Problèmes de dépendances** : Utilisez un environnement virtuel Python si nécessaire.
- **Port occupé** : Si le port 8501 est occupé, Streamlit utilisera automatiquement un autre port.

## Modèles utilisés

- **LLM** : qwen3:8b
- **Embeddings** : bge-m3
- **Reranker** : ms-marco-MultiBERT-L-12