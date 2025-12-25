import nltk
from cs336_data.extract_data import extract_texts_from_warc
import random

def run_gopher_quality_filter(text: str) -> bool:
    words = nltk.word_tokenize(text)
    num_words = len(words)
    if num_words < 50 or num_words > 100000:
        return False

    mean_word_length = sum(len(word) for word in words) / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    lines = text.splitlines()
    if len(lines) > 0:
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
        if (ellipsis_lines / len(lines)) > 0.3:
            return False

    alphabetic_words = sum(1 for word in words if any(c.isalpha() for c in word))
    if (alphabetic_words / num_words) < 0.8:
        return False

    return True

if __name__ == "__main__":

    warc_path = "CC-MAIN-20241201162023-20241201192023-00000.warc"
    texts = []
    texts = extract_texts_from_warc(warc_path)
    for i in range(10):
        print(f"{'='*60}")
        print(f"Document {i+1}")
        text = texts[i]
        if len(text) < 500:
            continue
        else:
            text_range = random.randint(0, len(text)-500)
            text = text[text_range:min(len(text), text_range+500)]
        print(text)
        is_high_quality = run_gopher_quality_filter(text)
        print(f"\nGopher Quality Filter: {'High Quality' if is_high_quality else 'Low Quality'}")