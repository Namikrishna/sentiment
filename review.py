import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------
# 1. Load and preprocess data
# ------------------------------------------
df = pd.read_csv('movie_review.csv')
print("Columns in CSV:", df.columns)

df.dropna(subset=['text', 'tag'], inplace=True)
df['label'] = df['tag'].str.lower().map({'pos': 1, 'neg': 0})

if df['label'].isnull().any():
    print("❌ Found unknown tag values:", df['tag'].unique())
    raise ValueError("Only 'pos' and 'neg' values allowed in 'tag' column.")

# ------------------------------------------
# 2. WordCloud Visualization
# ------------------------------------------
text_all = " ".join(df['text'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_all)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud of All Reviews", fontsize=16)
plt.show()

# ------------------------------------------
# 3. Train-test split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# ------------------------------------------
# 4. Bag-of-Words model
# ------------------------------------------
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

model_bow = LogisticRegression(max_iter=1000)
model_bow.fit(X_train_bow, y_train)
y_pred_bow = model_bow.predict(X_test_bow)

acc_bow = accuracy_score(y_test, y_pred_bow)
print("\n[BoW] Accuracy:", acc_bow)
print("Classification Report (BoW):\n", classification_report(y_test, y_pred_bow))

cm_bow = confusion_matrix(y_test, y_pred_bow)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_bow, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix - Bag of Words")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 5. TF-IDF model
# ------------------------------------------
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model_tfidf = LogisticRegression(max_iter=1000)  # fixed here
model_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

acc_tfidf = accuracy_score(y_test, y_pred_tfidf)
print("\n[TF-IDF] Accuracy:", acc_tfidf)
print("Classification Report (TF-IDF):\n", classification_report(y_test, y_pred_tfidf))

cm_tfidf = confusion_matrix(y_test, y_pred_tfidf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_tfidf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix - TF-IDF")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
