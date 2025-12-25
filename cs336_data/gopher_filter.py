import nltk

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