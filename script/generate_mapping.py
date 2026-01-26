"""
Script utilitaire pour générer le fichier url_mapping.json à partir des fichiers 
déjà téléchargés et du fichier uqac_gestion_links.txt.

À utiliser si vous avez déjà des fichiers téléchargés mais pas le mapping.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple


def url_to_name(url: str) -> str:
    """Génère un nom de fichier stable à partir de l'URL (SHA1)."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def read_links(links_path: Path) -> Tuple[List[str], List[str]]:
    """Lit et sépare les liens HTML et PDF depuis le fichier de liens."""
    html_links: List[str] = []
    pdf_links: List[str] = []
    current = None

    with links_path.open("r", encoding="utf-8") as f:
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


def generate_mapping():
    """Génère le fichier url_mapping.json à partir des liens."""
    # Chemins
    data_dir = Path(__file__).resolve().parent.parent / "data"
    links_path = data_dir / "uqac_gestion_links.txt"
    mapping_path = data_dir / "raw" / "url_mapping.json"
    
    if not links_path.exists():
        print(f"[ERREUR] Fichier de liens non trouvé : {links_path}")
        return
    
    # Lire les liens
    html_links, pdf_links = read_links(links_path)
    
    # Créer le mapping
    url_mapping: Dict[str, str] = {}
    
    for url in html_links:
        fname = url_to_name(url) + '.html'
        url_mapping[fname] = url
    
    for url in pdf_links:
        fname = url_to_name(url) + '.pdf'
        url_mapping[fname] = url
    
    # Sauvegarder le mapping
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(url_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Mapping généré : {len(url_mapping)} URLs")
    print(f"✓ Fichier créé : {mapping_path}")


if __name__ == "__main__":
    generate_mapping()
