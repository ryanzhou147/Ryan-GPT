import fasttext
import random
from cs336_data.extract_data import extract_texts_from_warc

model_path_nsfw = "jigsaw_fasttext_bigrams_nsfw_final.bin"
model_nsfw = fasttext.load_model(model_path_nsfw)

def classify_nsfw(string: str) -> tuple[str, float]:
    string = string.replace('\n', ' ')
    labels, scores = model_nsfw.predict(string, k=1)
    label = labels[0].replace('__label__', '')
    score = scores[0]
    return (label, score)

model_path_hatespeech = "jigsaw_fasttext_bigrams_hatespeech_final.bin"
model_hatespeech = fasttext.load_model(model_path_hatespeech)

def classify_hatespeech(string: str) -> tuple[str, float]:
    string = string.replace('\n', ' ')
    labels, scores = model_hatespeech.predict(string, k=1)
    label = labels[0].replace('__label__', '')
    score = scores[0]
    return (label, score)

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
        nsfw_label, nsfw_score = classify_nsfw(text)
        print(f"\nNSFW Prediction: {nsfw_label} (Confidence: {nsfw_score:.4f})")
        hatespeech_label, hatespeech_score = classify_hatespeech(text)
        print(f"\nHate Speech Prediction: {hatespeech_label} (Confidence: {hatespeech_score:.4f})")
