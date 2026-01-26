import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests


class RawDownloader:
    """Télécharge les fichiers HTML et PDF depuis les URLs listées."""

    # Paramètres réseau
    TIMEOUT = 15
    HEADERS = {"User-Agent": "Chatbot-UQAC/1.0"}
    SLEEP_RANGE_SEC = (1.0, 2.5)  # délai de politesse entre les requêtes

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialise le téléchargeur avec les chemins de travail."""
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.raw_html_dir = self.raw_dir / "html"
        self.raw_pdf_dir = self.raw_dir / "pdf"
        self.links_path = data_dir / "uqac_gestion_links.txt"
        self.mapping_path = self.raw_dir / "url_mapping.json"

    def ensure_dirs(self) -> None:
        """Crée les répertoires requis."""
        for folder in [self.raw_html_dir, self.raw_pdf_dir]:
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
            print(f"[CACHE] {dest.name} existe déjà")
            return dest
        try:
            print(f"[TÉLÉCHARGEMENT] {url}")
            resp = requests.get(url, timeout=self.TIMEOUT, headers=self.HEADERS)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[ERREUR] échec du téléchargement {url} : {exc}")
            return None
        self.polite_sleep()
        dest.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if isinstance(resp.content, (bytes, bytearray)) else "w"
        with dest.open(mode) as f:
            f.write(resp.content if mode == "wb" else resp.text)
        print(f"[OK] {dest.name} téléchargé")
        return dest

    def download_all(self) -> None:
        """Télécharge tous les fichiers HTML et PDF listés."""
        self.ensure_dirs()
        html_links, pdf_links = self.read_links()
        
        # Dictionnaire pour le mapping fichier -> URL
        url_mapping: Dict[str, str] = {}
        
        print(f"\n=== Téléchargement de {len(html_links)} pages HTML ===\n")
        html_count = 0
        for url in html_links:
            fname = self.url_to_name(url) + '.html'
            dest = self.raw_html_dir / fname
            if self.download(url, dest):
                html_count += 1
                url_mapping[fname] = url
        
        print(f"\n=== Téléchargement de {len(pdf_links)} fichiers PDF ===\n")
        pdf_count = 0
        for url in pdf_links:
            fname = self.url_to_name(url) + '.pdf'
            dest = self.raw_pdf_dir / fname
            if self.download(url, dest):
                pdf_count += 1
                url_mapping[fname] = url
        
        # Sauvegarder le mapping fichier -> URL
        with self.mapping_path.open("w", encoding="utf-8") as f:
            json.dump(url_mapping, f, ensure_ascii=False, indent=2)
        print(f"\nMapping URL sauvegardé : {self.mapping_path}")
        
        print(f"\n=== TERMINÉ ===")
        print(f"HTML: {html_count}/{len(html_links)} téléchargés")
        print(f"PDF: {pdf_count}/{len(pdf_links)} téléchargés")


if __name__ == "__main__":
    downloader = RawDownloader()
    downloader.download_all()
