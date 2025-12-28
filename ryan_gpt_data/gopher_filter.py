import re
from typing import Iterable, List

try:
    import nltk
    _NLTK_AVAILABLE = True
except Exception:
    nltk = None
    _NLTK_AVAILABLE = False


def _tokenize_words(text: str) -> List[str]:
    """Tokenize text into words. Prefer NLTK if available, otherwise use a regex fallback."""
    if _NLTK_AVAILABLE:
        try:
            return nltk.word_tokenize(text)
        except LookupError:
            # punkt not available — fall back
            pass

    # simple fallback tokenizer: words with letters/digits and internal hyphens/apostrophes
    return re.findall(r"[\w'-]+", text)


def run_gopher_quality_filter(text: str) -> bool:
    """Return True if text passes lightweight Gopher-style quality heuristics.

    The heuristics intentionally stay simple and fast so they can be run on large
    corpora without heavy NLP downloads.
    """
    words = _tokenize_words(text)
    num_words = len(words)
    if num_words < 50 or num_words > 100000:
        return False

    mean_word_length = sum(len(word) for word in words) / max(1, num_words)
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    lines = text.splitlines()
    if len(lines) > 0:
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
        if (ellipsis_lines / len(lines)) > 0.3:
            return False

    alphabetic_words = sum(1 for word in words if any(c.isalpha() for c in word))
    if (alphabetic_words / max(1, num_words)) < 0.8:
        return False

    return True


def filter_texts(texts: Iterable[str]) -> List[str]:
    """Return only the texts that pass the quality filter (preserves order)."""
    return [t for t in texts if run_gopher_quality_filter(t)]


if __name__ == "__main__":
    # Lightweight demo when run as a script
    from ryan_gpt_data.extract_data import extract_texts_from_warc
    import random

    warc_path = "CC-MAIN-20241201162023-20241201192023-00000.warc"
    texts = list(extract_texts_from_warc(warc_path))
    for i, text in enumerate(texts[:10]):
        print(f"{'='*60}")
        print(f"Document {i+1}")
        if len(text) < 500:
            print("(too short, skipped)")
            continue
        text_range = random.randint(0, max(0, len(text) - 500))
        snippet = text[text_range : text_range + 500]
        print(snippet)
        is_high_quality = run_gopher_quality_filter(snippet)
        print(f"\nGopher Quality Filter: {'High Quality' if is_high_quality else 'Low Quality'}")