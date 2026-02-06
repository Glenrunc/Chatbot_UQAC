import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pdfplumber
from bs4 import BeautifulSoup


class CorpusBuilder:
    """Traite les fichiers HTML/PDF bruts et crée le corpus avec chunking pour un chatbot RAG."""

    # Paramètres de découpage (chunking)
    CHUNK_SIZE_WORDS = 400
    CHUNK_OVERLAP_WORDS = 60

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
    def extract_html(path: Path) -> Tuple[str, str]:
        """Extrait le titre et le texte d'un fichier HTML depuis entry-header et entry-content"""
        text = ""
        title = ""
        raw = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "html.parser")
        
        # Extraire le titre de la balise <title>
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        
        # Extraire le texte des sections entry-header et entry-content
        text_parts = []
        
        for class_name in ["entry-header", "entry-content"]:
            element = soup.find(class_=class_name)
            if element:
                # Supprimer tous les liens mais garder leur texte
                for link in element.find_all("a"):
                    link.replace_with(link.get_text())
                text_parts.append(element.get_text(separator=" ", strip=True))
        
        # Combiner les parties et nettoyer
        text = " ".join(text_parts).strip()
        
        # Nettoyer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return title, text

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

    def _process_files(
        self, 
        files: List[Path],
        doc_type: str,
        extract_fn) -> List[dict]:
        """Traite une liste de fichiers locaux : extrait et découpe en chunks."""
        records = []
        for file_path in files:
            print(f"[TRAITEMENT] {file_path.name}")
            title, text = extract_fn(file_path)
            if not text:
                print(f"[AVERTISSEMENT] {doc_type.upper()} vide après extraction {file_path.name}")
                continue
            
            # Récupérer l'URL originale depuis le mapping, sinon utiliser un fallback
            url = self.url_mapping.get(file_path.name, f"local://{file_path.name}")
            
            for idx, chunk in enumerate(self.chunk_text(text)):
                rec = {
                    "id": f"{doc_type}-{file_path.name}-{idx}",
                    "url": url,
                    "type": doc_type,
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
        records.extend(self._process_files(html_files, "html", self.extract_html))
        
        print(f"\n=== Traitement de {len(pdf_files)} fichiers PDF ===\n")
        records.extend(self._process_files(pdf_files, "pdf", self.extract_pdf))

        self.write_jsonl(records)
        print(f"\n=== TERMINÉ ===")
        print(f"Corpus généré : {len(records)} segments -> {self.corpus_path}")


if __name__ == "__main__":
    builder = CorpusBuilder()
    builder.build()
