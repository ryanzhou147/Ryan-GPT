import fasttext
from ryan_gpt_data.extract_data import extract_texts_from_warc

model_path = "lid.176.bin"
model = fasttext.load_model(model_path)

def run_identify_language(text: str) -> tuple[str, float]:
    text = text.replace('\n', ' ')
    labels, scores = model.predict(text, k=1)
    lang = labels[0].replace('__label__', '')
    score = scores[0]
    return (lang, score)
    
if __name__ == "__main__":

    warc_path = "CC-MAIN-20241201162023-20241201192023-00000.warc"
    texts = []
    texts = extract_texts_from_warc(warc_path)
    for i in range(10):
        print(f"{'='*60}")
        print(f"Document {i+1}")
        text = texts[i]
        print(text[:100])
        lang, score = run_identify_language(text)
        print(f"\nPredicted Language: {lang} (Confidence: {score:.4f})")