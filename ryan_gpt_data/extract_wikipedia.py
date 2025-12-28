import regex as re
import os
import sys
import json
import shutil
import tempfile
import subprocess

def download_wikipedia(output_dir: str = "data/wikipedia"):
    """Download Wikipedia dump."""
    os.makedirs(output_dir, exist_ok=True)
    
    url = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
    output_path = f"{output_dir}/simplewiki-latest.xml.bz2"

    if not os.path.exists(output_path):
        print(f"Downloading Wikipedia dump from {url}...")
        subprocess.run(["wget", "-c", "-O", output_path, url], check=True)
    
    return output_path

RE_REF = re.compile(r'<ref[^>]*>.*?</ref>', flags=re.DOTALL)
RE_REF_SELF = re.compile(r'<ref[^/]*?/>')
RE_TEMPLATE = re.compile(r'\{\{[^}]+\}\}')
RE_CATEGORY = re.compile(r'\[\[Category:[^\]]+\]\]')
RE_LINK_PIPE = re.compile(r'\[\[[^|\]]+\|([^\]]+)\]\]')
RE_LINK = re.compile(r'\[\[([^\]]+)\]\]')
RE_EXTLINK = re.compile(r'\[https?://[^\]]+\]')
RE_HTML = re.compile(r'<[^>]+>')
RE_BOLD = re.compile(r"'{2,}")
RE_HEADING = re.compile(r'={2,}([^=]+)={2,}')
RE_NEWLINES = re.compile(r'\n{3,}')
RE_SPACES = re.compile(r' +')

RE_SECTION_HEADER = re.compile(r'^[A-Z][a-z]+\.\s*$', flags=re.MULTILINE)
RE_NUMBERED = re.compile(r'^\d+\)\s*', flags=re.MULTILINE)
RE_CHINESE = re.compile(r'\([^)]*[\u4e00-\u9fff][^)]*\)')
RE_CYRILLIC = re.compile(r'\([^)]*[\u0400-\u04FF][^)]*\)')
RE_PINYIN = re.compile(r'\(pinyin:[^)]+\)', flags=re.IGNORECASE)
RE_MULTI_SPACE = re.compile(r'  +')

def clean_wikitext(text: str) -> str:
    """Clean Wikipedia markup."""
    text = RE_REF.sub('', text)
    text = RE_REF_SELF.sub('', text)
    text = RE_TEMPLATE.sub('', text)
    text = RE_CATEGORY.sub('', text)
    text = RE_LINK_PIPE.sub(r'\1', text)
    text = RE_LINK.sub(r'\1', text)
    text = RE_EXTLINK.sub('', text)
    text = RE_HTML.sub('', text)
    text = RE_BOLD.sub('', text)
    text = RE_HEADING.sub(r'\1', text)
    text = RE_NEWLINES.sub('\n\n', text)
    text = RE_SPACES.sub(' ', text)
    return text.strip()


def clean_article(text: str) -> str:
    """Extra cleaning after initial clean_wikitext."""
    
    for marker in ['Related pages', 'References', 'Other websites', 'Notes', 'Sources']:
        if marker in text:
            text = text.split(marker)[0]
    
    text = RE_SECTION_HEADER.sub('', text)
    text = RE_NUMBERED.sub('', text)
    text = RE_CHINESE.sub('', text)
    text = RE_CYRILLIC.sub('', text)
    text = RE_PINYIN.sub('', text)
    text = RE_NEWLINES.sub('\n\n', text)
    text = RE_MULTI_SPACE.sub(' ', text)
    
    return text.strip()

def filter_article(text: str) -> bool:
    """Return True if article should be kept."""
    
    words = text.split()
    if len(words) < 100:
        return False
    
    lines = text.strip().split('\n')
    bullet_lines = sum(1 for line in lines if line.strip().startswith(('*', '-', '#')))
    if bullet_lines / max(len(lines), 1) > 0.5:
        return False
    
    # Faster ASCII check using encode
    try:
        ascii_ratio = len(text.encode('ascii', errors='ignore')) / max(len(text), 1)
        if ascii_ratio < 0.8:
            return False
    except:
        return False
    
    return True


def extract_wikipedia(
    dump_path: str,
    output_path: str = "data/wikipedia/wiki_text.txt",
    max_articles: int = None,
    min_length: int = 200,
    use_article_filter: bool = True,
    use_wikiextractor: bool = True,
    processes: int = 10,
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
        
        # Write article
        f_out.write(f"{title}")
        f_out.write(cleaned)
        f_out.write("\n<|endoftext|>\n")  
        
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
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    for root, _, files in os.walk(tempdir):
                        for fname in sorted(files):  # sorted for deterministic order
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

if __name__ == "__main__":
    dump_path = download_wikipedia()
    extract_wikipedia(
        dump_path,
        max_articles=20000,
        min_length=200,
        use_article_filter=True,
    )