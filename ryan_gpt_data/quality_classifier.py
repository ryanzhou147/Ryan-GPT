import fasttext
import random
import gzip
import subprocess
import glob
import os
from ryan_gpt_data.extract_data import extract_texts_from_warc

def classify_string(model: fasttext.FastText._FastText, text: str) -> tuple[str, float]:
    """Classify text using the provided fastText model."""
    text = ' '.join(text.split())
    labels, scores = model.predict(text, k=1)
    label = labels[0].replace('__label__', '')
    score = float(scores[0])
    return label, score

def sample_urls(urls_file: str, n: int = 1000, output_file: str = "sampled_urls.txt"):
    """Sample n random URLs from the file."""
    
    # Skip if already done
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            count = sum(1 for _ in f)
        print(f"[CACHED] {output_file} already exists ({count} URLs)")
        return output_file
    
    print(f"Reading URLs from {urls_file}...")
    
    urls = []
    with gzip.open(urls_file, 'rt', errors='ignore') as f:
        for line in f:
            url = line.strip()
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
    """Scrape URLs in parallel using background processes."""
    
    # Check if WARC files already exist
    existing_warcs = glob.glob(f"{warc_prefix}_chunk_*.warc.gz")
    if existing_warcs:
        print(f"[CACHED] Found {len(existing_warcs)} existing WARC files")
        return f"{warc_prefix}_chunk_*.warc.gz"
    
    print(f"Scraping URLs from {urls_file} with {jobs} parallel jobs...")
    
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # Split into chunks
    chunk_size = max(1, len(urls) // jobs)
    chunks = [urls[i:i + chunk_size] for i in range(0, len(urls), chunk_size)]
    
    # Run wget in parallel
    processes = []
    for i, chunk in enumerate(chunks):
        chunk_file = f"{warc_prefix}_chunk_{i}.txt"
        with open(chunk_file, 'w') as f:
            f.write('\n'.join(chunk))
        
        cmd = [
            "wget",
            "--timeout=3",
            "--tries=1",
            "-i", chunk_file,
            f"--warc-file={warc_prefix}_chunk_{i}",
            "-O", "/dev/null",
            "--quiet",
            "--no-check-certificate",
        ]
        processes.append(subprocess.Popen(cmd))
    
    # Wait for all
    for i, p in enumerate(processes):
        p.wait()
        print(f"  Chunk {i+1}/{len(processes)} done")
    
    # Cleanup temp .txt files
    for i in range(len(chunks)):
        chunk_file = f"{warc_prefix}_chunk_{i}.txt"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
    
    return f"{warc_prefix}_chunk_*.warc.gz"


def extract_from_warcs(warc_pattern: str, max_docs: int = 5000) -> list[str]:
    """Extract texts from multiple WARC files matching pattern."""
    
    all_texts = []
    
    # Expand glob pattern
    warc_files = glob.glob(warc_pattern)
    
    if not warc_files:
        print(f"Warning: No files found matching {warc_pattern}")
        return []
    
    print(f"Found {len(warc_files)} WARC files")
    
    for warc_path in warc_files:
        print(f"  Extracting from {warc_path}...")
        try:
            texts = extract_texts_from_warc(warc_path)
            for text in texts:
                if len(text.split()) > 100:
                    all_texts.append(text)
                if len(all_texts) >= max_docs:
                    break
        except Exception as e:
            print(f"    Error: {e}")
        
        if len(all_texts) >= max_docs:
            break
    
    print(f"Extracted {len(all_texts)} documents total")
    return all_texts


def prepare_fasttext_data(
    positive_texts: list[str], 
    negative_texts: list[str], 
    output_file: str = "quality_train.txt"
) -> str:
    """Create fastText training file."""
    
    # Skip if already done
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            count = sum(1 for _ in f)
        print(f"[CACHED] {output_file} already exists ({count} examples)")
        return output_file
    
    def clean(text):
        text = ' '.join(text.split())
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
    
    # Skip if already done
    if os.path.exists(model_path):
        print(f"[CACHED] {model_path} already exists")
        return fasttext.load_model(model_path)
    
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
    print(f"Training - Samples: {n}, Precision: {p:.4f}, Recall: {r:.4f}")
    
    return model


MODEL_PATH = "quality_classifier.bin"
_model = None


def run_classify_quality(text: str) -> tuple[bool, float]:
    """Classify text quality. Returns: (is_high_quality, confidence)"""
    global _model
    if _model is None:
        _model = fasttext.load_model(MODEL_PATH)

    text = ' '.join(text.split())
    labels, scores = _model.predict(text, k=1)
    
    label = labels[0].replace('__label__', '')
    score = float(scores[0])
    
    return (label == "high_quality", score)


if __name__ == "__main__":
    random.seed(42)
    
    CC_WARC = "CC-MAIN-20241201162023-20241201192023-00000.warc"
    WIKI_URLS = "enwiki-20240420-extracted_urls.txt.gz"
    SAMPLED_URLS = 1000
    PARALLEL_JOBS = 10
    
    print("STEP 1: Sample Wikipedia URLs")
    sampled_urls = sample_urls(WIKI_URLS, n=SAMPLED_URLS, output_file="sampled_wiki_urls.txt")
    
    print("STEP 2: Scrape URLs in parallel")
    warc_pattern = scrape_urls_parallel(sampled_urls, "positive_samples", jobs=PARALLEL_JOBS)
    
    print("STEP 3: Extract positive examples from Wikipedia URLs")
    positive_texts = extract_from_warcs(warc_pattern)
    
    print("STEP 4: Extract negative examples from Common Crawl")
    negative_texts = extract_texts_from_warc(CC_WARC)
    negative_texts = [t for t in negative_texts if len(t.split()) > 100]
    
    n_samples = min(len(positive_texts), len(negative_texts))
    positive_texts = positive_texts[:n_samples]
    negative_texts = negative_texts[:n_samples]
    print(f"Balanced: {n_samples} positive, {n_samples} negative")
    
    print("STEP 5: Prepare training data")
    train_file = prepare_fasttext_data(positive_texts, negative_texts)
    
    print("STEP 6: Train classifier")
    model = train_classifier(train_file)
    
    print("STEP 7: Test classifier")
    test_cases = [
        ("The French Revolution began in 1789 with the convocation of the Estates-General. "
         "The first year saw the Tennis Court Oath, assault on the Bastille, and the Declaration of Rights. "
         "This period marked a fundamental transformation in European political history.",
         "HIGH"),
        ("BUY NOW!!! CLICK HERE for amazing deals FREE SHIPPING!!! Make $$$ from home! "
         "Limited time offer! Act now! Subscribe for updates! Best prices guaranteed!!!",
         "LOW"),
    ]
    
    for text, expected in test_cases:
        is_high, score = run_classify_quality(text)
        predicted = "HIGH" if is_high else "LOW"
        print(f"Expected: {expected}, Got: {predicted} ({score:.4f})")
    
    print("Model saved to: quality_classifier.bin")

    temp_files = glob.glob("positive_samples_chunk_*.warc.gz") + glob.glob("positive_samples_chunk_*.txt")
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)