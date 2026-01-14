import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

EXCLUDED_PATTERNS = [
    'wp-admin',
    'wp-login',
    '?action=',
    '?post=',
    '?p=',
    '?preview=',
    '?replytocom=',
    '?share=',
    '/feed/',
    '/trackback/',
    '/comment-page-',
]

def is_valid_url(url: str, base_url: str) -> bool:
    if not url or not url.startswith(base_url):
        return False
    
    for pattern in EXCLUDED_PATTERNS:
        if pattern in url:
            return False
    
    return True

def is_valid_pdf(url: str, base_url: str) -> bool:
    if not url or not url.lower().endswith('.pdf'):
        return False
    if not url.startswith(base_url):
        return False
    return True

def crawler_uqac(base_url: str, save=False) -> tuple[list[str], list[str]]:
    """
    Crawler récursif pour explorer tous les liens du site UQAC https://www.uqac.ca/mgestion/.
    
    Args:
        base_url: URL de départ pour le crawling
        save: Si True, sauvegarde les liens dans un fichier
        
    Returns:
        Tuple contenant (liens_html, liens_pdf) triés
    """
    visited = set()
    to_visit = {base_url}
    links_html = set()
    links_pdf = set()
    
    while to_visit:
        current_url = to_visit.pop()
        
        if current_url in visited:
            continue
            
        visited.add(current_url)
        
        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Erreur lors de l'accès à {current_url}: {e}")
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for link in soup.find_all('a'):
            href = link.get('href')
            
            if not href:
                continue
            
            href = href.split('#')[0].strip()
            
            if not href:
                continue
            
            if not href.startswith('http'):
                href = urljoin(current_url, href)
            
            if is_valid_pdf(href, base_url):
                links_pdf.add(href)
                continue
            
            if is_valid_url(href, base_url) and href not in visited:
                links_html.add(href)
                to_visit.add(href)
        
        print(f"Visité: {len(visited)} | À visiter: {len(to_visit)} | HTML: {len(links_html)} | PDF: {len(links_pdf)}", end='\r')

    
    if save:
        with open("../data/uqac_gestion_links.txt", "w") as f:
            f.write("# Pages HTML\n")
            for link in sorted(links_html):
                f.write(f"{link}\n")
            f.write("\n# Fichiers PDF\n")
            for link in sorted(links_pdf):
                f.write(f"{link}\n")
    
    return sorted(links_html), sorted(links_pdf)


if __name__ == "__main__":
    start_url = "https://www.uqac.ca/mgestion/"
    links_html, links_pdf = crawler_uqac(start_url, save=True)