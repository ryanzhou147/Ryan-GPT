
def mask_pii(text: str, pattern: str = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}") -> tuple[str, int]:
    import re

    masked_text, num_subs = re.subn(pattern, "|||EMAIL_ADDRESS|||", text)
    return masked_text, num_subs

