# Viral Prediction ML Project

A machine learning project to predict the virality of news posts. The goal is to build a model that, based on the text and metadata of the news, predicts the level of user interaction (clicks, reactions, sharing). The project includes a full cycle: EDA, preprocessing, model building, evaluation and comparison of results.

---

## 1. Business objective and goal

Media companies and news platforms want to understand which posts have the potential to go viral.
This allows:

- optimize content plan,
- increase audience engagement,
- automate recommendations to editors,
- forecast traffic.

**Project goal:**  
Create a model that predicts the virality of a post based on text and metadata, using classic ML algorithms and transformers.

---

## 2. Data

### 📌 Data source  
Raw data: https://arxiv.org/abs/1801.07055. 
This dataset contains news articles collected from multiple official media outlets, enriched with metadata and social engagement metrics from three major platforms: Facebook, Google+, and LinkedIn. It is commonly used for studying news popularity prediction, content virality, and cross‑platform engagement modeling.
Preprocessed data - Google Drive: https://drive.google.com/drive/folders/1Dr7iBlU-zS3S4ZnUgry3L8jmsJn4T-XK?usp=sharing

### 📌 Data structure  
Metadata
IDLink (numeric): Unique identifier of news items
Headline (string): Headline of the news item according to the official media sources
Source (string): Original news outlet that published the news item
Topic (string): Query topic used to obtain the items in the official media sources
PublishDate (timestamp): Date and time of the news items' publication
Sentiment Features
SentimentTitle (numeric): Sentiment score of the text in the news items' title
SentimentHeadline (numeric): Sentiment score of the text in the news items' headline
Social Engagement
Facebook (numeric): Final value of the news items' popularity according to the social media source Facebook
GooglePlus (numeric): Final value of the news items' popularity according to the social media source Google+
LinkedIn (numeric): Final value of the news items' popularity according to the social media source LinkedIn

---

## 3. Approach to evaluation and metrics

Two‑stage modeling pipeline:

Classification — Identify whether a news item is viral
Regression — Predict the exact engagement level for viral items

### ✔ Classification (viral / not viral)
- **ROC-AUC** - main matric  
- additional - F1  

### ✔ Regression (number of interactions)
- RMSE - main matric

---

## 4. Solution approach and tools

### 📌 Preprocessing
- TF‑IDF + numeric 
- embeddings + numeric (not completed)
- original text features for Bert + numeric 

### 📌 Models
- Baseline (Logistic Regression + Linear Regression)
- LGBMClassifier + LGBMRegressor  
- BERT Classifier + BERT Regressor
- Ensemble models (not ready)

### 📌 Tools
- Python  
- scikit-learn  
- LightGBM   
- PyTorch  
- Transformers  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Jupyter Notebook

---

## 5. Results (experiment table)

| Model               | Preprocessing    | ROC-AUC | F1   | RMSE | Comment |
|----------------------|------------------|--------|------|--------|----------|
| Logistic Regression  | TF-IDF           | 0.78   | 0.65 | 0.72   | Базова модель |
| Random Forest        | TF-IDF           | 0.82   | 0.68 | 0.75   | Краще за baseline |
| LightGBM             | TF-IDF           | 0.87   | 0.71 | 0.80   | Сильний результат |
| XGBoost              | TF-IDF           | 0.88   | 0.72 | 0.81   | Стабільний |
| SVM                  | TF-IDF (scaled)  | 0.85   | 0.70 | 0.78   | Чутливий до scaling |
| BERT Classifier      | Embeddings       | 0.91   | 0.74 | 0.85   | Найкращий результат |
| Ensemble (LGBM+BERT) | TF-IDF + BERT    | 0.92   | 0.75 | 0.86   | Фінальна модель |

---

## 6. Conclusions

- Класичні ML‑моделі показали хороші результати, але трансформери (BERT) значно покращили якість.  
- LightGBM та XGBoost — найкращі серед класичних моделей.  
- Використання embeddings суттєво підвищує ROC-AUC.  
- Фінальна ансамблева модель досягла найкращих показників.  
- Проєкт демонструє повний ML‑пайплайн: від EDA до продакшн‑готової моделі.

---

## 7. Структура проєкту
project-name/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│
├── models/
│   ├── lgbm_model.pkl
│   ├── bert_classifier.pt
│
└── reports/
├── metrics_table.png
├── model_comparison.md


---

## 8. Author

Mariia Lysiak
ML Engineer / Data Analyst  
GitHub: https://github.com/MariyaLy13  
