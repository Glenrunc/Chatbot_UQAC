from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


class UQACCrawler:
    """Crawl récursif du site UQAC (manuel de gestion) pour collecter les liens HTML et PDF."""

    EXCLUDED_PATTERNS = [
        "wp-admin",
        "wp-login",
        "?action=",
        "?post=",
        "?p=",
        "?preview=",
        "?replytocom=",
        "?share=",
        "/feed/",
        "/trackback/",
        "/comment-page-",
    ]

    TIMEOUT = 10
    HEADERS = {"User-Agent": "Chatbot-UQAC/1.0"}
    SLEEP_RANGE_SEC = (0.8, 1.6)

    def __init__(self, base_url: str = "https://www.uqac.ca/mgestion/", links_path: Path | None = None) -> None:
        """Initialise le crawler avec l'URL de base et le chemin de sauvegarde des liens."""
        self.base_url = base_url.rstrip("/") + "/"
        if links_path is None:
            links_path = Path(__file__).resolve().parent.parent / "data" / "uqac_gestion_links.txt"
        self.links_path = links_path

    def _is_valid_url(self, url: str) -> bool:
        if not url or not url.startswith(self.base_url):
            return False
        for pattern in self.EXCLUDED_PATTERNS:
            if pattern in url:
                return False
        return True

    def _is_valid_pdf(self, url: str) -> bool:
        return bool(url and url.lower().endswith(".pdf") and url.startswith(self.base_url))

    def _polite_sleep(self) -> None:
        # Délai de politesse pour ne pas surcharger le site cible.
        import random
        import time

        time.sleep(random.uniform(*self.SLEEP_RANGE_SEC))

    def crawl(self, save: bool = False) -> tuple[list[str], list[str]]:
        """Explore récursivement le site de base et retourne les liens HTML/PDF trouvés."""
        visited: set[str] = set()
        to_visit: set[str] = {self.base_url}
        links_html: set[str] = set()
        links_pdf: set[str] = set()

        while to_visit:
            current_url = to_visit.pop()
            if current_url in visited:
                continue
            visited.add(current_url)

            try:
                response = requests.get(current_url, timeout=self.TIMEOUT, headers=self.HEADERS)
                response.raise_for_status()
            except requests.RequestException as exc:
                print(f"[AVERTISSEMENT] accès impossible {current_url} : {exc}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            for link in soup.find_all("a"):
                href = link.get("href")
                if not href:
                    continue
                href = href.split("#")[0].strip()
                if not href:
                    continue
                if not href.startswith("http"):
                    href = urljoin(current_url, href)

                if self._is_valid_pdf(href):
                    links_pdf.add(href)
                    continue

                if self._is_valid_url(href) and href not in visited:
                    links_html.add(href)
                    to_visit.add(href)

            print(
                f"Visité: {len(visited)} | À visiter: {len(to_visit)} | HTML: {len(links_html)} | PDF: {len(links_pdf)}",
                end="\r",
            )

            self._polite_sleep()

        if save:
            self._save_links(links_html, links_pdf)

        return sorted(links_html), sorted(links_pdf)

    def _save_links(self, links_html: set[str], links_pdf: set[str]) -> None:
        """Sauvegarde les liens collectés dans le fichier de sortie."""
        self.links_path.parent.mkdir(parents=True, exist_ok=True)
        with self.links_path.open("w", encoding="utf-8") as f:
            f.write("# Pages HTML\n")
            for link in sorted(links_html):
                f.write(f"{link}\n")
            f.write("\n# Fichiers PDF\n")
            for link in sorted(links_pdf):
                f.write(f"{link}\n")


if __name__ == "__main__":
    crawler = UQACCrawler()
    crawler.crawl(save=True)