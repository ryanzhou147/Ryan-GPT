# Problem (gopher_quality_filters): 3 points
# (a) Implement (at least) the subset of the Gopher quality filters as described above. For tokenizing
# text into words, you might find the NLTK package useful (specifically nltk.word_tokenize),
# though you’re not required to use it.
# Deliverable: A function that takes a string as its only argument and returns a boolean indicating whether the text passes the Gopher quality filters. Implement the adapter
# [run_gopher_quality_filter]. Then, make sure your filters pass the tests in uv run pytest
# -k test_gopher.
# (b) Run your rule-based quality filter on text extracted from the WARC files (via your previouslyimplemented text extraction function). Look through 20 random examples and compare the filter
# predictions to your own judgment. Comment on any cases where the quality filters differ from
# your judgments.
# Deliverable: A 2-5 sentence response.

# Contain less than 50 or more than 100,000 words.
# • Have a mean word length outside the range of 3 to 10 characters.
# • Have more than 30% of lines ending with an ellipsis (“...”).
# • Contain less than 80% of words with at least one alphabetic character

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