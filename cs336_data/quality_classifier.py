import fasttext
import random
import gzip
import subprocess
from cs336_data.extract_data import extract_texts_from_warc


def sample_urls(urls_file: str, n: int = 1000, output_file: str = "sampled_urls.txt"):
    """Sample n random URLs from the file."""
    print(f"Reading URLs from {urls_file}...")
    
    urls = []
    with gzip.open(urls_file, 'rt', errors='ignore') as f:
        for line in f:
            url = line.strip()
            # Filter: only http(s), no spaces, reasonable length
            if url.startswith('http') and ' ' not in url and len(url) < 500:
                urls.append(url)
    
    print(f"Total valid URLs: {len(urls):,}")
    
    sampled = random.sample(urls, min(n, len(urls)))
    
    with open(output_file, 'w') as f:
        for url in sampled:
            f.write(url + '\n')
    
    print(f"Sampled {len(sampled)} URLs to {output_file}")
    return output_file

def scrape_urls_parallel(urls_file: str, warc_prefix: str = "positive_samples", jobs: int = 10) -> str:
    """Scrape URLs in parallel using xargs."""
    
    cmd = f"cat {urls_file} | xargs -P {jobs} -I {{}} wget --timeout=3 --tries=1 -q -O /dev/null {{}}"
    subprocess.run(cmd, shell=True, check=False)
    return f"{warc_prefix}.warc.gz"

def extract_from_warc(warc_path: str, max_docs: int = 5000) -> list[str]:
    texts = extract_texts_from_warc(warc_path)
    filtered = []
    for text in texts:
        if 100 < len(text.split()) < max_docs:
            filtered.append(text)

    print(f"Extracted {len(filtered)} documents")
    return filtered

def prepare_fasttext_data(positive_texts: list[str], negative_texts: list[str], output_file: str = "quality_train.txt") -> str:
    """Create fastText training file."""
    
    def clean(text):
        # FastText needs single line per example
        text = ' '.join(text.split())
        # Truncate
        return ' '.join(text.split()[:300])
    
    lines = []
    
    for text in positive_texts:
        clean_text = clean(text)
        if len(clean_text.split()) > 30:
            lines.append(f"__label__high_quality {clean_text}")
    
    for text in negative_texts:
        clean_text = clean(text)
        if len(clean_text.split()) > 30:
            lines.append(f"__label__low_quality {clean_text}")
    
    random.shuffle(lines)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Created {output_file} with {len(lines)} examples")
    return output_file


def train_classifier(train_file: str, model_path: str = "quality_classifier.bin"):
    """Train fastText classifier."""
    
    print("Training fastText classifier...")
    
    model = fasttext.train_supervised(
        input=train_file,
        epoch=25,
        lr=0.1,
        wordNgrams=2,
        dim=100,
        loss='softmax',
    )
    
    model.save_model(model_path)
    
    n, p, r = model.test(train_file)
    print(f"Training - Precision: {p:.4f}, Recall: {r:.4f}")
    
    return model


MODEL_PATH = "quality_classifier.bin"
try:
    _model = fasttext.load_model(MODEL_PATH)
except Exception:
    _model = None


def run_classify_quality(text: str) -> tuple[bool, float]:
    """
    Classify text quality.
    Returns: (is_high_quality, confidence)
    """
    global _model
    if _model is None:
        _model = fasttext.load_model(MODEL_PATH)

    text = ' '.join(text.split())  # Clean whitespace
    labels, scores = _model.predict(text, k=1)
    
    label = labels[0].replace('__label__', '')
    score = float(scores[0])
    
    is_high_quality = (label == "high_quality")
    
    return (is_high_quality, score)

if __name__ == "__main__":
    random.seed(42)
    
    CC_WARC = "CC-MAIN-20241201162023-20241201192023-00000.warc"
    SAMPLED_URLS = 3000
    
    print("STEP 1: Getting positive examples from Wikipedia URLs")
    wiki_urls = "enwiki-20240420-extracted_urls.txt.gz"
    sampled_urls = sample_urls(wiki_urls, n=SAMPLED_URLS, output_file="sampled_wiki_urls.txt")
    positive_warc = scrape_urls_parallel(sampled_urls, "positive_samples", jobs=10)
    positive_texts = extract_from_warc(positive_warc)
    
    print("STEP 2: Getting negative examples from Common Crawl")
    negative_texts = extract_from_warc(CC_WARC, max_docs=3000)
    
    print("STEP 3: Preparing training data")
    train_file = prepare_fasttext_data(positive_texts, negative_texts)
    
    print("STEP 4: Training classifier")
    model = train_classifier(train_file)
    
    print("STEP 5: Testing")
    test_cases = [
        ("The French Revolution began in 1789 with the convocation of the Estates-General in May. "
         "The first year of the Revolution saw members of the Third Estate proclaiming the Tennis Court Oath, "
         "the assault on the Bastille, the passage of the Declaration of the Rights of Man and of the Citizen.", 
         "Expected: HIGH"),
        
        ("BUY NOW!!! CLICK HERE for amazing deals FREE SHIPPING!!! "
         "Make $$$ from home! Limited time offer! Act now! Subscribe for updates!",
         "Expected: LOW"),
    ]
    
    for text, expected in test_cases:
        is_high, score = run_classify_quality(text)
        label = "HIGH" if is_high else "LOW"
        print(f"\n{expected}")
        print(f"Predicted: {label} (confidence: {score:.4f})")
        print(f"Text: {text[:80]}...")
    