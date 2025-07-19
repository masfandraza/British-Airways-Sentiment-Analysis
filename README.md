# British-Airways-Sentiment-Analysis (2016–2023)
This project analyzes British Airways passenger reviews from 2016 to 2023 to understand customer sentiment and identify service improvement opportunities. Using natural language processing (NLP) and machine learning, it classifies whether a customer would recommend British Airways based on their written feedback.

---

## Overview

- **Data Source**: Kaggle – British Airways Passenger Reviews  
- **Techniques Used**: Text cleaning, TF-IDF vectorization, SMOTE, Ensemble modeling  
- **Goal**: Predict `Recommended` or `Not Recommended` and extract actionable service insights

---

## Key Features

- Cleaned and transformed 600+ real-world airline reviews
- Labeled sentiment using **TextBlob**: Positive, Neutral, Negative
- Balanced the target classes using **SMOTE**
- Combined models using a **Stacking Classifier**:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Logistic Regression (as final estimator)
- Visualized evaluation with:
  - Confusion Matrix
  - ROC Curve (AUC = **0.97**)
  - Precision-Recall Curve

---

## Model Results

| Metric       | Value     |
|--------------|-----------|
| Accuracy     | 93%       |
| Precision    | 0.93      |
| Recall       | 0.94      |
| F1 Score     | High      |
| AUC-ROC      | 0.97      |

The model demonstrates strong performance in classifying passenger recommendation behavior based on textual reviews.

---

## 📈usiness Insights

- **Economy Class** had the most neutral/negative reviews → needs service improvement
- **Positive sentiment** strongly correlated with higher star ratings and likelihood to recommend
- Some customers gave 4-star ratings but wrote neutral/negative reviews → highlights importance of text over rating alone
- **Regional satisfaction** varied by seat type (e.g., Premium Economy rated highly in Egypt)

---

## Tech Stack

- Python
- pandas, scikit-learn, XGBoost, imbalanced-learn
- TextBlob for sentiment scoring
- Matplotlib & Seaborn for visualization

---

## Files Included

- `Sentiment Analysis.py` – Full ML pipeline code
- `British_Airway_Review.csv` – Dataset used for this project
- `README.md` – Project summary and usage

---

## How to Run

1. Clone the repo  
2. Place the dataset (`British_Airway_Review.csv`) in the root directory  
3. Run `Sentiment Analysis.py` in Jupyter or your preferred IDE

---

