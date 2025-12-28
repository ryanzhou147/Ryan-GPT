from fastwarc.warc import ArchiveIterator, WarcRecordType

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str:
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.encoding import detect_encoding

    try:
        html_str = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding, errors='replace')
    return extract_plain_text(html_str)

def extract_texts_from_warc(warc_path: str) -> list[str]:
    texts = []
    with open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f):
            # Use record.record_type for fastwarc
            if record.record_type == WarcRecordType.response:
                # Use record.reader.read() for fastwarc
                content = record.reader.read()
                
                # Skip HTTP headers (separated by \r\n\r\n)
                if b'\r\n\r\n' in content:
                    html_bytes = content.split(b'\r\n\r\n', 1)[1]
                else:
                    html_bytes = content
                
                if html_bytes:
                    text = run_extract_text_from_html_bytes(html_bytes)
                    texts.append(text)
    return texts


def extract_texts_from_wet(wet_path: str) -> list[str]:
    texts = []
    with open(wet_path, 'rb') as f:
        for record in ArchiveIterator(f):
            if record.record_type == WarcRecordType.conversion:
                text = record.reader.read().decode('utf-8', errors='replace')
                texts.append(text)
    return texts


if __name__ == "__main__":
    warc_path = "CC-MAIN-20241201162023-20241201192023-00000.warc"
    wet_path = "CC-MAIN-20241201162023-20241201192023-00000.warc.wet"
    
    warc_texts = extract_texts_from_warc(warc_path)
    wet_texts = extract_texts_from_wet(wet_path)
    
    for i in range(2):
        print(f"{'='*60}")
        print(f"Document {i+1}")
        print(f"--- Our Extraction (WARC) ---")
        print(warc_texts[i][:500])
        print(f"\n--- Common Crawl (WET) ---")
        print(wet_texts[i][:500])