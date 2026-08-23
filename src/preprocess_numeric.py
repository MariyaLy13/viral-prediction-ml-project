import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_numeric(df):
    df = df.copy()

    # Ensure text columns are strings
    df["Title"] = df["Title"].fillna("").astype(str)
    df["Headline"] = df["Headline"].fillna("").astype(str)
    df["Source"] = df["Source"].fillna("").astype(str)
    
    # Sentiment cleanup
    df[['SentimentTitle', 'SentimentHeadline']] = \
        df[['SentimentTitle', 'SentimentHeadline']].fillna(
            df[['SentimentTitle', 'SentimentHeadline']].median()
        )

    # Engagement cleanup
    df[['Facebook', 'GooglePlus', 'LinkedIn']] = \
        df[['Facebook', 'GooglePlus', 'LinkedIn']].replace(-1, 0)

    # Date features
    df['PublishDate'] = pd.to_datetime(df['PublishDate'], errors='coerce')
    df['year'] = df['PublishDate'].dt.year
    df['month'] = df['PublishDate'].dt.month
    df['day'] = df['PublishDate'].dt.day
    df['weekday'] = df['PublishDate'].dt.weekday
    df['hour'] = df['PublishDate'].dt.hour

    # Topic one-hot
    df = pd.get_dummies(df, columns=['Topic'], drop_first=True)

    # Length features
    df["title_length_chars"] = df["Title"].apply(lambda x: len(str(x)))
    df["title_length_words"] = df["Title"].apply(lambda x: len(str(x).split()))

    df["headline_length_chars"] = df["Headline"].apply(lambda x: len(str(x)))
    df["headline_length_words"] = df["Headline"].apply(lambda x: len(str(x).split()))

    df["headline_punctuation_count"] = df["Headline"].apply(
    lambda x: len(re.findall(r"[!?.,]", str(x)))
    )

    df["headline_uppercase_ratio"] = df["Headline"].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
    )

    # Viral column
    df["engagement_total"] = df["Facebook"] + df["GooglePlus"] + df["LinkedIn"]
    viral_threshold = df["engagement_total"].quantile(0.90)
    df["is_viral"] = (df["engagement_total"] >= viral_threshold).astype(int)

    # Numeric columns
    num_cols = [
        'SentimentTitle', 'SentimentHeadline',
        'year', 'month', 'day', 'weekday', 'hour',
        'title_length_chars', 'title_length_words',
        'headline_length_chars', 'headline_length_words',
        'headline_punctuation_count', 'headline_uppercase_ratio'
    ] + [col for col in df.columns if col.startswith('Topic')]

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols])

    return df, X_num, num_cols, scaler