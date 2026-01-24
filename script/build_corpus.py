import hashlib
import json
import random
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import requests
import trafilatura
import pdfplumber
from bs4 import BeautifulSoup


class CorpusBuilder:
    """Extraction, découpage (chunking) et préparation du corpus UQAC pour un chatbot RAG."""

    # Paramètres de découpage (chunking)
    CHUNK_SIZE_WORDS = 400
    CHUNK_OVERLAP_WORDS = 60

    # Paramètres réseau
    TIMEOUT = 15
    HEADERS = {"User-Agent": "Chatbot-UQAC/1.0"}
    SLEEP_RANGE_SEC = (1.0, 2.5)  # délai de politesse entre les requêtes

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
        self.links_path = data_dir / "uqac_gestion_links.txt"

    def ensure_dirs(self) -> None:
        """Crée les répertoires requis."""
        for folder in [self.raw_html_dir, self.raw_pdf_dir, self.processed_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def read_links(self) -> Tuple[List[str], List[str]]:
        """Lit et sépare les liens HTML et PDF depuis le fichier de liens."""
        html_links: List[str] = []
        pdf_links: List[str] = []
        current = None

        with self.links_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    if "Pages HTML" in line:
                        current = "html"
                    elif "Fichiers PDF" in line:
                        current = "pdf"
                    continue
                if current == "html":
                    html_links.append(line)
                elif current == "pdf":
                    pdf_links.append(line)
        return html_links, pdf_links

    @staticmethod
    def url_to_name(url: str) -> str:
        """Génère un nom de fichier stable à partir de l'URL (SHA1)."""
        return hashlib.sha1(url.encode("utf-8")).hexdigest()

    def polite_sleep(self) -> None:
        """Attend un court délai aléatoire pour éviter de surcharger le site distant."""
        time.sleep(random.uniform(*self.SLEEP_RANGE_SEC))

    def download(self, url: str, dest: Path) -> Path | None:
        """Télécharge l'URL vers le chemin cible. Retourne None en cas d'échec; utilise le cache si déjà présent."""
        if dest.exists():
            return dest
        try:
            resp = requests.get(url, timeout=self.TIMEOUT, headers=self.HEADERS)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[AVERTISSEMENT] échec du téléchargement {url} : {exc}")
            return None
        self.polite_sleep()
        dest.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if isinstance(resp.content, (bytes, bytearray)) else "w"
        with dest.open(mode) as f:
            f.write(resp.content if mode == "wb" else resp.text)
        return dest

    @staticmethod
    def extract_html(path: Path) -> Tuple[str, str]:
        """Extrait le titre et le texte d'un fichier HTML"""
        text = ""
        title = ""
        raw = path.read_text(encoding="utf-8", errors="ignore")
        extracted = trafilatura.extract(raw, include_comments=False, include_tables=True)
        if extracted:
            text = extracted.strip()
        soup = BeautifulSoup(raw, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
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

    def _process_links(
        self, 
        urls: List[str],
        doc_type: str,
        dest_dir: Path,
        extract_fn) -> List[dict]:
        """Traite une liste d'URLs : télécharge, extrait et découpe en chunks."""
        records = []
        for url in urls:
            fname = self.url_to_name(url) + f'.{doc_type}'
            dest = dest_dir / fname
            saved = self.download(url, dest)
            if not saved:
                continue
            title, text = extract_fn(saved)
            if not text:
                print(f"[AVERTISSEMENT] {doc_type.upper()} vide après extraction {url}")
                continue
            for idx, chunk in enumerate(self.chunk_text(text)):
                rec = {
                    "id": f"{doc_type}-{fname}-{idx}",
                    "url": url,
                    "type": doc_type,
                    "title": title,
                    "chunk_id": idx,
                    "chunk": chunk,
                }
                records.append(rec)
        return records

    def build(self) -> None:
        """Construit le corpus : télécharge, extrait, découpe et écrit tous les documents."""
        self.ensure_dirs()
        html_links, pdf_links = self.read_links()
        
        records = []
        records.extend(self._process_links(html_links, "html", self.raw_html_dir, self.extract_html))
        records.extend(self._process_links(pdf_links, "pdf", self.raw_pdf_dir, self.extract_pdf))

        self.write_jsonl(records)
        print(f"Corpus généré : {len(records)} segments -> {self.corpus_path}")


if __name__ == "__main__":
    builder = CorpusBuilder()
    builder.build()
