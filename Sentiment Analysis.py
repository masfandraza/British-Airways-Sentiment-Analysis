import pandas as pd
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import shap
import numpy as np
import warnings

# Suppress warnings for XGBoost
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# Load the dataset
file_path = 'British_Airway_Review.csv'  # Replace with the correct file path
df = pd.read_csv(file_path)

# Data Preprocessing
# Handle missing values
df.fillna({'reviews': '', 'country': 'Unknown', 'seat_type': 'Unknown', 'type_of_traveller': 'Unknown', 'date': '01 January 2000'}, inplace=True)

# Clean 'reviews' column by removing unnecessary symbols/emojis and converting to lowercase
df['cleaned_reviews'] = df['reviews'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))

# Convert 'date' to datetime format
def clean_date(date_str):
    date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
    return pd.to_datetime(date_str, format='%d %B %Y', errors='coerce')

df['date_cleaned'] = df['date'].apply(clean_date)

# Sentiment Analysis using TextBlob to classify as Positive, Neutral, or Negative
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['cleaned_reviews'].apply(get_sentiment)

# Add sentiment polarity score as a new feature
df['sentiment_score'] = df['cleaned_reviews'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Features and target selection
X_text = df['cleaned_reviews']
X_cat = df[['seat_type', 'type_of_traveller', 'country']]
y = df['recommended']

# TF-IDF Vectorization for text data
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# OneHotEncoding for categorical features
one_hot = OneHotEncoder(drop='first', sparse_output=False)

# ColumnTransformer to handle different types of features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', tfidf, 'cleaned_reviews'),
        ('cat', one_hot, ['seat_type', 'type_of_traveller', 'country'])
    ])

# Data Preprocessing Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_processed = pipeline.fit_transform(df)

# Label Encoding the target
y_encoded = LabelEncoder().fit_transform(y)

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y_encoded)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define base models for stacking
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
]

# Define the meta-classifier
meta_classifier = LogisticRegression()

# Define the Stacking Classifier
stacking_model = StackingClassifier(estimators=base_estimators, final_estimator=meta_classifier, cv=5)

# Train the Stacking Classifier
stacking_model.fit(X_train, y_train)

# Predict on the test set with the stacking model
y_pred_stack = stacking_model.predict(X_test)
y_pred_proba_stack = stacking_model.predict_proba(X_test)[:, 1]

# Evaluate the stacking model
conf_matrix_stack = confusion_matrix(y_test, y_pred_stack)
class_report_stack = classification_report(y_test, y_pred_stack, target_names=['Not Recommended', 'Recommended'])
roc_auc = roc_auc_score(y_test, y_pred_proba_stack)

# Display the confusion matrix for the stacking model
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_stack, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Recommended', 'Recommended'], yticklabels=['Not Recommended', 'Recommended'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Stacking Classifier Recommendation Prediction')
plt.show()

# Print the classification report for the stacking model
print("Classification Report for Stacking Classifier Model:\n")
print(class_report_stack)

# Print AUC-ROC score
print(f"AUC-ROC Score for Stacking Classifier Model: {roc_auc:.2f}")

# Plot AUC-ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_stack)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC Curve for Stacking Classifier')
plt.legend(loc='lower right')
plt.show()

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_stack)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Stacking Classifier')
plt.show()

