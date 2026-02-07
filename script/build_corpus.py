import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import httpx
import pdfplumber
from bs4 import BeautifulSoup
from tqdm import tqdm


class CorpusBuilder:
    """Traite les fichiers HTML/PDF bruts et crée le corpus avec chunking pour un chatbot RAG."""

    # Paramètres de découpage (chunking)
    CHUNK_SIZE_WORDS = 400
    CHUNK_OVERLAP_WORDS = 60
    
    # Configuration Ollama pour embeddings
    OLLAMA_BASE_URL = "http://localhost:11434"
    EMBEDDING_MODEL = "bge-m3"
    EMBEDDING_BATCH_SIZE = 32  # Nombre de chunks à embedder en parallèle

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialise le constructeur de corpus avec les chemins de travail."""
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.raw_html_dir = self.raw_dir / "html"
        self.raw_pdf_dir = self.raw_dir / "pdf"
        self.processed_dir = data_dir / "processed"
        self.corpus_path = self.processed_dir / "corpus.jsonl"
        self.mapping_path = self.raw_dir / "url_mapping.json"
        self.url_mapping: Dict[str, str] = {}

    def ensure_dirs(self) -> None:
        """Crée les répertoires requis."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_url_mapping(self) -> None:
        """Charge le mapping fichier -> URL depuis le fichier JSON."""
        if self.mapping_path.exists():
            with self.mapping_path.open("r", encoding="utf-8") as f:
                self.url_mapping = json.load(f)
            print(f"Mapping chargé : {len(self.url_mapping)} URLs")
        else:
            print(f"[AVERTISSEMENT] Fichier de mapping non trouvé : {self.mapping_path}")
            print("Les URLs seront au format 'local://nom_fichier'")


    @staticmethod
    def extract_html(path: Path) -> Tuple[str, List[dict]]:
        """
        Extrait le titre et des sections structurées d'un fichier HTML.
        
        Retourne le titre principal et une liste de sections avec leur hiérarchie.
        Chaque section = {"heading": str, "level": int, "content": str}
        """
        raw = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "html.parser")
        
        # Extraire le titre principal
        title = ""
        entry_header = soup.find(class_="entry-header")
        if entry_header:
            h1 = entry_header.find("h1")
            if h1:
                title = h1.get_text(strip=True)
        
        # Fallback sur <title> si pas de h1
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()
            # Nettoyer le suffixe " | Manuel de gestion" souvent présent
            if " | " in title:
                title = title.split(" | ")[0].strip()
        
        sections: List[dict] = []
        entry_content = soup.find(class_="entry-content")
        
        if not entry_content:
            return title, sections
        
        # Supprimer les scripts et styles
        for tag in entry_content.find_all(["script", "style"]):
            tag.decompose()
        
        # Supprimer les liens mais garder leur texte
        for link in entry_content.find_all("a"):
            link.replace_with(link.get_text())
        
        # Identifier tous les éléments enfants directs et les headings
        heading_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
        
        current_heading = title  # Section par défaut = titre principal
        current_level = 0
        current_content: List[str] = []
        
        def get_heading_level(tag_name: str) -> int:
            """Retourne le niveau du heading (1-6) ou 0 si pas un heading."""
            if tag_name in heading_tags:
                return int(tag_name[1])
            return 0
        
        def clean_text(text: str) -> str:
            """Nettoie le texte des espaces multiples."""
            return re.sub(r'\s+', ' ', text).strip()
        
        def save_section():
            """Sauvegarde la section courante si elle a du contenu."""
            if current_content:
                content_text = clean_text(" ".join(current_content))
                if content_text:
                    sections.append({
                        "heading": current_heading,
                        "level": current_level,
                        "content": content_text
                    })
        
        # Parcourir tous les éléments du contenu
        for element in entry_content.children:
            if not hasattr(element, 'name') or element.name is None:
                # Texte brut
                text = str(element).strip()
                if text:
                    current_content.append(text)
                continue
            
            level = get_heading_level(element.name)
            
            if level > 0:
                # C'est un heading - sauvegarder la section précédente
                save_section()
                
                # Nouvelle section
                current_heading = clean_text(element.get_text())
                current_level = level
                current_content = []
            else:
                # Contenu normal - l'ajouter à la section courante
                text = element.get_text(separator=" ", strip=True)
                if text:
                    current_content.append(text)
        
        # Sauvegarder la dernière section
        save_section()
        
        return title, sections

    @staticmethod
    def extract_pdf(path: Path) -> Tuple[str, str]:
        """Extrait le titre et le texte d'un fichier PDF"""
        title = path.stem
        parts: List[str] = []
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    parts.append(page_text.strip())
        except Exception as exc:
            print(f"[AVERTISSEMENT] échec de l'extraction PDF {path.name} : {exc}")
            return title, ""
        return title, "\n\n".join(p for p in parts if p)

    def chunk_text(self, text: str) -> List[str]:
        """Découpe le texte en segments (chunks) de mots avec recouvrement."""
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = min(len(words), start + self.CHUNK_SIZE_WORDS)
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start = max(0, end - self.CHUNK_OVERLAP_WORDS)
        return chunks

    def chunk_sections(self, title: str, sections: List[dict]) -> List[dict]:
        """
        Découpe les sections HTML en chunks optimisés pour le RAG.
        
        Stratégie:
        - Les petites sections sont fusionnées avec contexte du heading parent
        - Les grandes sections sont découpées avec overlap
        - Chaque chunk conserve le contexte hiérarchique (titre + section)
        """
        chunks: List[dict] = []
        
        if not sections:
            return chunks
        
        # Construire la hiérarchie des headings pour le contexte
        # Format: [(level, heading), ...]
        heading_stack: List[Tuple[int, str]] = []
        
        for section in sections:
            heading = section["heading"]
            level = section["level"]
            content = section["content"]
            
            # Mettre à jour la stack des headings
            # Retirer tous les headings de niveau >= au niveau actuel
            if level > 0:
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading))
            
            # Construire le contexte hiérarchique (uniquement les headings, pas les niveaux)
            if heading_stack:
                context_path = " > ".join(h[1] for h in heading_stack)
            else:
                context_path = heading
            
            # Compter les mots du contenu
            words = content.split()
            word_count = len(words)
            
            if word_count == 0:
                continue
            
            # Si le contenu est assez petit, créer un seul chunk avec contexte
            if word_count <= self.CHUNK_SIZE_WORDS:
                chunk_text = f"[{title}] {context_path}\n\n{content}"
                chunks.append({
                    "heading": heading,
                    "context": context_path,
                    "text": chunk_text
                })
            else:
                # Découper les grandes sections avec overlap
                start = 0
                sub_idx = 0
                while start < word_count:
                    end = min(word_count, start + self.CHUNK_SIZE_WORDS)
                    chunk_content = " ".join(words[start:end])
                    
                    # Ajouter le contexte
                    if sub_idx > 0:
                        chunk_text = f"[{title}] {context_path} (suite)\n\n{chunk_content}"
                    else:
                        chunk_text = f"[{title}] {context_path}\n\n{chunk_content}"
                    
                    chunks.append({
                        "heading": heading,
                        "context": context_path,
                        "text": chunk_text
                    })
                    
                    if end == word_count:
                        break
                    start = max(0, end - self.CHUNK_OVERLAP_WORDS)
                    sub_idx += 1
        
        return chunks

    def write_jsonl(self, records: Iterable[dict]) -> None:
        """Écrit les enregistrements au format JSONL dans le fichier de corpus."""
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with self.corpus_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def get_local_files(self) -> Tuple[List[Path], List[Path]]:
        """Récupère la liste des fichiers HTML et PDF déjà téléchargés."""
        html_files = sorted(self.raw_html_dir.glob("*.html"))
        pdf_files = sorted(self.raw_pdf_dir.glob("*.pdf"))
        return html_files, pdf_files

    def _process_html_files(self, files: List[Path]) -> List[dict]:
        """Traite les fichiers HTML avec chunking sémantique basé sur les titres."""
        records = []
        for file_path in files:
            print(f"[TRAITEMENT] {file_path.name}")
            title, sections = self.extract_html(file_path)
            
            if not sections:
                print(f"[AVERTISSEMENT] HTML vide après extraction {file_path.name}")
                continue
            
            # Récupérer l'URL originale depuis le mapping
            url = self.url_mapping.get(file_path.name, f"local://{file_path.name}")
            
            # Chunking sémantique basé sur les sections
            chunks = self.chunk_sections(title, sections)
            
            for idx, chunk_data in enumerate(chunks):
                rec = {
                    "id": f"html-{file_path.name}-{idx}",
                    "url": url,
                    "type": "html",
                    "title": title,
                    "section": chunk_data["heading"],
                    "context": chunk_data["context"],
                    "chunk_id": idx,
                    "chunk": chunk_data["text"],
                }
                records.append(rec)
        
        return records

    def _process_pdf_files(self, files: List[Path]) -> List[dict]:
        """Traite les fichiers PDF avec chunking classique."""
        records = []
        for file_path in files:
            print(f"[TRAITEMENT] {file_path.name}")
            title, text = self.extract_pdf(file_path)
            
            if not text:
                print(f"[AVERTISSEMENT] PDF vide après extraction {file_path.name}")
                continue
            
            # Récupérer l'URL originale depuis le mapping
            url = self.url_mapping.get(file_path.name, f"local://{file_path.name}")
            
            for idx, chunk in enumerate(self.chunk_text(text)):
                rec = {
                    "id": f"pdf-{file_path.name}-{idx}",
                    "url": url,
                    "type": "pdf",
                    "title": title,
                    "chunk_id": idx,
                    "chunk": chunk,
                }
                records.append(rec)
        
        return records

    def build(self) -> None:
        """Construit le corpus à partir des fichiers HTML et PDF déjà téléchargés."""
        self.ensure_dirs()
        self.load_url_mapping()
        html_files, pdf_files = self.get_local_files()
        
        print(f"\n=== Traitement de {len(html_files)} fichiers HTML ===\n")
        records = []
        records.extend(self._process_html_files(html_files))
        
        print(f"\n=== Traitement de {len(pdf_files)} fichiers PDF ===\n")
        records.extend(self._process_pdf_files(pdf_files))

        # Générer les embeddings
        print(f"\n=== Génération des embeddings ({self.EMBEDDING_MODEL}) ===")
        records = self.add_embeddings(records)

        self.write_jsonl(records)
        print(f"\n=== TERMINÉ ===")
        print(f"Corpus généré : {len(records)} segments -> {self.corpus_path}")

    def get_embedding(self, text: str) -> List[float]:
        """Obtient l'embedding d'un texte via Ollama."""
        response = httpx.post(
            f"{self.OLLAMA_BASE_URL}/api/embed",
            json={"model": self.EMBEDDING_MODEL, "input": text},
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Obtient les embeddings de plusieurs textes en une seule requête."""
        response = httpx.post(
            f"{self.OLLAMA_BASE_URL}/api/embed",
            json={"model": self.EMBEDDING_MODEL, "input": texts},
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def add_embeddings(self, records: List[dict]) -> List[dict]:
        """Ajoute les embeddings à tous les enregistrements par batch."""
        print(f"Génération de {len(records)} embeddings...")
        
        # Traiter par batches pour plus d'efficacité
        for i in tqdm(range(0, len(records), self.EMBEDDING_BATCH_SIZE), desc="Embeddings"):
            batch = records[i:i + self.EMBEDDING_BATCH_SIZE]
            texts = [rec["chunk"] for rec in batch]
            
            try:
                embeddings = self.get_embeddings_batch(texts)
                for rec, emb in zip(batch, embeddings):
                    rec["embedding"] = emb
            except Exception as e:
                print(f"\n[ERREUR] Batch {i//self.EMBEDDING_BATCH_SIZE}: {e}")
                # Fallback: traiter un par un
                for rec in batch:
                    try:
                        rec["embedding"] = self.get_embedding(rec["chunk"])
                    except Exception as e2:
                        print(f"[ERREUR] Embedding pour {rec['id']}: {e2}")
                        rec["embedding"] = []
        
        return records


if __name__ == "__main__":
    builder = CorpusBuilder()
    builder.build()
