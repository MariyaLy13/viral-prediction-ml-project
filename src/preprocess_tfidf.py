import re
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_tfidf(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_tfidf(df):
    tfidf_title = TfidfVectorizer(max_features=3000, ngram_range=(1,2), stop_words='english')
    tfidf_headline = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    tfidf_source = TfidfVectorizer(max_features=1000, stop_words='english')

    X_title = tfidf_title.fit_transform(df['Title'].apply(clean_tfidf))
    X_headline = tfidf_headline.fit_transform(df['Headline'].apply(clean_tfidf))
    X_source = tfidf_source.fit_transform(df['Source'].apply(clean_tfidf))

    X_tfidf = sp.hstack([X_title, X_headline, X_source])

    return X_tfidf, tfidf_title, tfidf_headline, tfidf_source