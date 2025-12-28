import re
from ryan_gpt_data.extract_data import extract_texts_from_warc
import random

def mask_email(text: str, pattern: str = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}") -> tuple[str, int]:

    masked_text, num_subs = re.subn(pattern, "|||EMAIL_ADDRESS|||", text)
    return masked_text, num_subs

def mask_phone_number(text: str, pattern: str = r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}") -> tuple[str, int]:
    masked_text, num_subs = re.subn(pattern, "|||PHONE_NUMBER|||", text)
    return masked_text, num_subs

def mask_ip_address(text: str, pattern: str = r"\b(?:\d{1,3}\.){3}\d{1,3}\b") -> tuple[str, int]:
    masked_text, num_subs = re.subn(pattern, "|||IP_ADDRESS|||", text)
    return masked_text, num_subs

if __name__ == "__main__":
    warc_path = "CC-MAIN-20241201162023-20241201192023-00000.warc"
    texts = []
    texts = extract_texts_from_warc(warc_path)

    for i in range(10):
        print(f"{'='*60}")
        print(f"Document {i+1}")
        text = texts[i]

        for i in range(10):
            text, email_count = mask_email(text)
            text, phone_count = mask_phone_number(text)
            text, ip_count = mask_ip_address(text)
        print(text[random.randint(0, len(text)-500):min(len(text), random.randint(0, len(text)-500)+500)])
        print(f"\nMasked {email_count} emails, {phone_count} phone numbers, {ip_count} IP addresses.")