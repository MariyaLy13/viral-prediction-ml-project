import re

def clean_st(text):
    text = text.strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    return text