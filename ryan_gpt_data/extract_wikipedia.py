import bz2
import xml.etree.ElementTree as ET
import re
import os
from pathlib import Path

def download_wikipedia(output_dir: str = "data/wikipedia"):
    """Download Wikipedia dump."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Latest English Wikipedia dump (articles only)
    url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    output_path = f"{output_dir}/enwiki-latest.xml.bz2"
    
    if not os.path.exists(output_path):
        print(f"Downloading Wikipedia dump (~20GB)...")
        import subprocess
        subprocess.run(["wget", "-O", output_path, url], check=True)
    
    return output_path


def clean_wikitext(text: str) -> str:
    """Clean Wikipedia markup."""
    
    # Remove references
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*?/>', '', text)
    
    # Remove templates {{...}}
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    
    # Remove categories [[Category:...]]
    text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)
    
    # Convert links [[link|text]] to text
    text = re.sub(r'\[\[[^|\]]+\|([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    
    # Remove external links [http://...]
    text = re.sub(r'\[https?://[^\]]+\]', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove bold/italic markers
    text = re.sub(r"'{2,}", '', text)
    
    # Remove headings markup
    text = re.sub(r'={2,}([^=]+)={2,}', r'\1', text)
    
    # Clean whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def extract_wikipedia(
    dump_path: str,
    output_path: str = "data/wikipedia/wiki_text.txt",
    max_articles: int = None,
    min_length: int = 500,
):
    """Extract and clean text from Wikipedia dump."""
    
    print(f"Extracting Wikipedia from {dump_path}...")
    
    article_count = 0
    
    with bz2.open(dump_path, 'rt', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        # Parse XML incrementally
        context = ET.iterparse(f_in, events=('end',))
        
        for event, elem in context:
            if elem.tag.endswith('page'):
                # Extract title and text
                title = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}title')
                text = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}text')
                
                if title is not None and text is not None and text.text:
                    # Skip non-article pages
                    if ':' in title.text:  # Wikipedia:, Template:, etc.
                        elem.clear()
                        continue
                    
                    cleaned = clean_wikitext(text.text)
                    
                    if len(cleaned) >= min_length:
                        # Write article
                        f_out.write(f"# {title.text}\n\n")
                        f_out.write(cleaned)
                        f_out.write("\n\n")
                        
                        article_count += 1
                        
                        if article_count % 10000 == 0:
                            print(f"  Extracted {article_count} articles...")
                        
                        if max_articles and article_count >= max_articles:
                            break
                
                # Clear element to save memory
                elem.clear()
    
    print(f"Extracted {article_count} articles to {output_path}")
    return output_path


if __name__ == "__main__":
    dump_path = download_wikipedia()
    extract_wikipedia(dump_path, max_articles=100000)  # Start with 100k for testing