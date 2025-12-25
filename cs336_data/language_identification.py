# Write a function that will take a Unicode string and identify the main language that is present
# in this string. Your function should return a pair, containing an identifier of the language and a
# score between 0 and 1 representing its confidence in that prediction.
# Deliverable: A function that performs language identification, giving its top language prediction
# and a score. Implement the adapter [run_identify_language] and make sure it passes both
# tests in uv run pytest -k test_identify_language . Note that these tests assume a particular
# string identifier for English (“en”) and Chinese (“zh”), so your test adapter should perform any
# applicable re-mapping, if necessary.

import fasttext
import os

model_path = "lid.176.bin"
model = fasttext.load_model(model_path)

def run_identify_language(text: str) -> tuple[str, float]:
    text = text.replace('\n', ' ')
    labels, scores = model.predict(text, k=1)
    lang = labels[0].replace('__label__', '')
    score = scores[0]
    return (lang, score)
    
if __name__ == "__main__":
    sample_text_en = "This is a sample English text."
    sample_text_zh = "这是一个中文示例文本。"
    
    print(run_identify_language(sample_text_en))  # Expected output: ('en', score)
    print(run_identify_language(sample_text_zh))  # Expected output: ('zh', score)