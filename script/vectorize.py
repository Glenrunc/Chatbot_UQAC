import os
import shutil
import requests
from bs4 import BeautifulSoup

from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DB_DIR = "../data/chroma_db"
LINKS_FILE = "../data/uqac_gestion_links.txt"

def scrapper_html_semantic(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # parser
        soup = BeautifulSoup(response.content, 'lxml')
        main_content = soup.find("div", class_="entry-content")
        
        if not main_content:
            print(f"Pas de contenu 'entry-content' trouvé sur {url}")
            return []

        # Récupération du titre principal
        page_title = "Document UQAC"
        h1_tag = soup.find("h1", class_="entry-title")
        if h1_tag:
            page_title = h1_tag.get_text(strip=True)

        # Définition de la hiérarchie pour le découpage
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
        ]

        html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)
        html_string = str(main_content)
        
        html_header_splits = html_splitter.split_text(html_string)

        final_docs = []
        for doc in html_header_splits:
            doc.metadata["source"] = url
            doc.metadata["main_title"] = page_title
            if not doc.page_content.strip():
                continue
            final_docs.append(doc)
            
        return final_docs

    except Exception as e:
        print(f"Erreur sur {url}: {e}")
        return []

def main():
    if os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
        except Exception as e:
            print(f"Impossible de supprimer le dossier DB (peut-être ouvert ?) : {e}")

    all_documents = []
    
    # Vérification que le fichier de liens existe
    if not os.path.exists(LINKS_FILE):
        print(f"ERREUR : Le fichier {LINKS_FILE} est introuvable")
        return

    with open(LINKS_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in urls:
        docs = scrapper_html_semantic(url)
        all_documents.extend(docs)

    print(f"Nombre de sections trouvées : {len(all_documents)}")

    #Si aucun document, on arrête tout
    if len(all_documents) == 0:
        print("ARRÊT : Aucun document n'a été récupéré. Vérifiez vos liens ou votre connexion.")
        return

    # Découpage de sécurité
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    final_splits = text_splitter.split_documents(all_documents)
    print(f"Nombre final de chunks : {len(final_splits)}")

    # Injection de contexte
    for split in final_splits:
        main_title = split.metadata.get("main_title", "Document")
        h1 = split.metadata.get("Header 1", "")
        h2 = split.metadata.get("Header 2", "")
        source = split.metadata.get("source", "")
        
        structure = f"{main_title}"
        if h1: structure += f" > {h1}"
        if h2: structure += f" > {h2}"
        
        header_block = f"DOCUMENT : {structure}\nSOURCE : {source}\nCONTENU :\n"
        split.page_content = header_block + split.page_content

    # Vectorisation
    print("Création de la base vectorielle...")
    embeddings = OllamaEmbeddings(model="llama3")
    
    # On ajoute try/except pour voir l'erreur exacte si Chroma plante
    try:
        vectorstore = Chroma.from_documents(
            documents=final_splits,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        print("Terminé")
    except Exception as e:
        print(f"Erreur ChromaDB : {e}")

if __name__ == "__main__":
    main()
