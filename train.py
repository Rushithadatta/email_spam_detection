import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle



df = pd.read_csv("spam_ham_dataset.csv", encoding='latin-1')
# print(df.columns)
df = df[['text','label_num']]
print(df.isnull().sum())

# print(df['label_num'].value_counts())
# counts = df['label_num'].value_counts()

# plt.bar(['Ham','Spam'], counts)
# plt.title("Spam vs Ham Distribution")
# plt.show()



nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['cleaned_text'] = df['text'].apply(clean_text)
# print(df[['text','cleaned_text']].head())

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label_num']

print(X.shape)
# print(vectorizer.get_feature_names_out()[:20])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)
report = classification_report(y_test, predictions)
print("Classification Report:")
print(report)


pickle.dump(model, open("spam_model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))