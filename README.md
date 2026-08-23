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
- Raw data: https://arxiv.org/abs/1801.07055. 
This dataset contains news articles collected from multiple official media outlets, enriched with metadata and social engagement metrics from three major platforms: Facebook, Google+, and LinkedIn. It is commonly used for studying news popularity prediction, content virality, and cross‑platform engagement modeling.
- Preprocessed data - Google Drive: https://drive.google.com/drive/folders/1Dr7iBlU-zS3S4ZnUgry3L8jmsJn4T-XK?usp=sharing

### 📌 Data structure  
- Metadata
- IDLink (numeric): Unique identifier of news items
- Headline (string): Headline of the news item according to the official media sources
- Source (string): Original news outlet that published the news item
- Topic (string): Query topic used to obtain the items in the official media sources
- PublishDate (timestamp): Date and time of the news items' publication
- Sentiment Features
- SentimentTitle (numeric): Sentiment score of the text in the news items' title
- SentimentHeadline (numeric): Sentiment score of the text in the news items' headline
- Social Engagement
- Facebook (numeric): Final value of the news items' popularity according to the social media source Facebook
- GooglePlus (numeric): Final value of the news items' popularity according to the social media source Google+
- LinkedIn (numeric): Final value of the news items' popularity according to the social media source LinkedIn

---

## 3. Approach to evaluation and metrics

Two‑stage modeling pipeline:

- **Classification** — Identify whether a news item is viral
- **Regression** — Predict the exact engagement level for viral items

### 📌 Classification (viral / not viral)
- **ROC-AUC** - main matric - the best choice for ranking quality: show how well the model separates viral posts from non-viral ones, regardless of the classification threshold.This is critical for the virality prediction task, where businesses are not just interested in determining a class, but rather in understanding which posts have the highest viral potential. ROC-AUC is a robust metric in the presence of class imbalance.
- additional - F1  

### 📌 Regression (number of interactions)
- RMSE - main matric - RMSE shows the average error of the model in units of the target (number of interactions) and strongly penalizes large errors, which is important for predicting virality. The metric is easily interpreted by businesses and is a standard in tasks of predicting numerical values ​​(traffic, sales, interactions).

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

| Model                | Split            | ROC-AUC |   F1   |     RMSE      |
|----------------------|------------------|---------|--------|---------------|
| Baseline             | Train            | 0.9300  | 0.4580 | 2,192,539.35  |
| Baseline             | Val              | 0.8670  | 0.3281 | 8,444.59      |
| Baseline             | Test             | 0.8684  | 0.3296 | 77,707,033.05 |
| LightGBM             | Train            | 0.9992  | 0.9072 | 441.56        |
| LightGBM             | Val              | 0.8700  | 0.3293 | 530.34        |
| LightGBM             | Test             | 0.8678  | 0.3228 | 595.26        |
| BERT                 | Train            | 0.5020  | 0.0960 | 726.42        |
| BERT                 | Val              | 0.8608  | 0.4254 | 564.28        |
| BERT                 | Test             | 0.8618  | 0.4364 | 547.04        |
| Hybrid approach      | pending          | 0.0000  | 0.0000 | 0.0000        |

---

## 6. Conclusions

The experiment results suggest that no single model fully captures both the structural metadata
patterns (where LightGBM excels) and the deep semantic patterns in text (where BERT excels). This
motivates exploring hybrid approaches.

### 7. Future Work

Several promising directions can further improve the system:

1. **Two‑Stage Hybrid Pipeline (LGBM Classifier + BERT Regressor)**  
   A combined approach where LightGBM first identifies potentially viral posts, and BERT then
   predicts the expected engagement level for those posts. This mirrors real editorial workflows:
   first detect “high‑potential” content, then estimate its impact. This pipeline can reduce noise,
   improve ranking quality, and provide more actionable predictions.

2. **Improved Text Representations**  
   Using domain‑specific transformer models (e.g., DistilBERT fine‑tuned on news data) may improve
   classification performance. Ligther model then Bert give me the opportunity to train throuth more iteration to improve ROC AUC on train for classifier

---

## 8. Author

Mariia Lysiak
ML Engineer / Data Analyst  
GitHub: https://github.com/MariyaLy13  
