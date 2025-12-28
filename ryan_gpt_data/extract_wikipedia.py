import bz2
import xml.etree.ElementTree as ET
import re
import os
import sys
import json
import shutil
import tempfile
import subprocess


def download_wikipedia(output_dir: str = "data/wikipedia", use_simple: bool = True):
    """Download Wikipedia dump."""
    os.makedirs(output_dir, exist_ok=True)
    
    if use_simple:
        url = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
        output_path = f"{output_dir}/simplewiki-latest.xml.bz2"
    else:
        url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
        output_path = f"{output_dir}/enwiki-latest.xml.bz2"
    
    if not os.path.exists(output_path):
        print(f"Downloading Wikipedia dump from {url}...")
        subprocess.run(["wget", "-c", "-O", output_path, url], check=True)
    
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


def filter_article(text: str) -> bool:
    """Return True if article should be kept."""
    
    # 1. Minimum word count (skip stubs)
    if len(text.split()) < 100:
        return False
    
    # 2. Skip list-heavy articles (more than 50% bullet lines)
    lines = text.strip().split('\n')
    bullet_lines = sum(1 for line in lines if line.strip().startswith(('*', '-', '#')))
    if bullet_lines / max(len(lines), 1) > 0.5:
        return False
    
    # 3. Skip if too many section markers (little real content)
    if text.count('Related pages') + text.count('References') > 2:
        return False
    
    return True


def clean_article(text: str) -> str:
    """Extra cleaning after initial clean_wikitext."""
    
    # Remove "Related pages" section and everything after
    for marker in ['Related pages', 'References', 'Other websites', 'Notes', 'Sources']:
        if marker in text:
            text = text.split(marker)[0]
    
    # Remove section headers (keep just the content)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove numbered list formatting but keep content
    text = re.sub(r'^\d+\)\s*', '', text, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return text


def extract_wikipedia(
    dump_path: str,
    output_path: str = "data/wikipedia/wiki_text.txt",
    max_articles: int = None,
    min_length: int = 300,
    use_gopher_filter: bool = False,
    use_article_filter: bool = True,
    use_wikiextractor: bool = False,
    processes: int = 4,
):
    """Extract and clean text from Wikipedia dump."""
    
    print(f"Extracting Wikipedia from {dump_path}...")
    
    # Namespace for this dump
    NS = "{http://www.mediawiki.org/xml/export-0.11/}"
    
    article_count = 0
    skipped_count = 0
    
    # Helper function to process a single article
    def process_article(title: str, text: str, f_out) -> bool:
        nonlocal article_count, skipped_count
        
        # Skip non-article pages
        if ':' in title:
            return False
        
        # Clean text
        cleaned = clean_wikitext(text)
        cleaned = clean_article(cleaned)
        
        # Length filter
        if len(cleaned) < min_length:
            skipped_count += 1
            return False
        
        # Article quality filter
        if use_article_filter and not filter_article(cleaned):
            skipped_count += 1
            return False
        
        # Gopher filter (optional)
        if use_gopher_filter:
            try:
                from ryan_gpt_data.gopher_filter import run_gopher_quality_filter
            except Exception:
                run_gopher_quality_filter = None
            
            if run_gopher_quality_filter is not None and not run_gopher_quality_filter(cleaned):
                skipped_count += 1
                return False
        
        # Write article
        f_out.write(f"{title}\n\n")
        f_out.write(cleaned)
        f_out.write("\n\n---\n\n")
        
        article_count += 1
        
        if article_count % 10000 == 0:
            print(f"  Extracted {article_count} articles (skipped {skipped_count})...")
        
        return True
    
    # If requested, use WikiExtractor for much faster, parallel extraction.
    if use_wikiextractor:
        tempdir = tempfile.mkdtemp(prefix="wikiex-")
        wikiex_failed = False
        
        try:
            cmd = [
                sys.executable, '-m', 'wikiextractor.WikiExtractor',
                '--json', '-o', tempdir, '--processes', str(processes), dump_path
            ]
            print('Running WikiExtractor:', ' '.join(cmd))
            
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print('WikiExtractor failed, falling back to XML parser:', e)
                shutil.rmtree(tempdir, ignore_errors=True)
                wikiex_failed = True
            
            if not wikiex_failed:
                # Walk the output directory and read JSON lines
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    for root, _, files in os.walk(tempdir):
                        for fname in files:
                            fpath = os.path.join(root, fname)
                            with open(fpath, 'r', encoding='utf-8') as fh:
                                for line in fh:
                                    try:
                                        obj = json.loads(line)
                                    except Exception:
                                        continue
                                    
                                    title = obj.get('title') or ''
                                    text = obj.get('text') or ''
                                    
                                    process_article(title, text, f_out)
                                    
                                    if max_articles and article_count >= max_articles:
                                        print(f"Extracted {article_count} articles (skipped {skipped_count}) to {output_path}")
                                        return output_path
                
                print(f"Extracted {article_count} articles (skipped {skipped_count}) to {output_path}")
                return output_path
        
        finally:
            try:
                shutil.rmtree(tempdir)
            except Exception:
                pass
        
        if wikiex_failed:
            # Reset counts and fall through to XML parser
            article_count = 0
            skipped_count = 0
    
    # Fallback: original streaming XML parser (single-threaded)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with bz2.open(dump_path, 'rt', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        context = ET.iterparse(f_in, events=('end',))
        
        for event, elem in context:
            if elem.tag == f"{NS}page":
                
                title_elem = elem.find(f"{NS}title")
                text_elem = elem.find(f".//{NS}text")
                
                if title_elem is not None and text_elem is not None and text_elem.text:
                    title = title_elem.text
                    text = text_elem.text
                    
                    process_article(title, text, f_out)
                    
                    if max_articles and article_count >= max_articles:
                        break
                
                elem.clear()
    
    print(f"Extracted {article_count} articles (skipped {skipped_count}) to {output_path}")
    return output_path


if __name__ == "__main__":
    dump_path = download_wikipedia(use_simple=True)
    extract_wikipedia(
        dump_path,
        max_articles=20000,
        min_length=300,
        use_article_filter=True,
        use_gopher_filter=False,
    )