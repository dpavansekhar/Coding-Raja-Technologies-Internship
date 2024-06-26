from flask import Flask, request, render_template
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the model and vectorizer
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

stop_words = set(stopwords.words('english'))

def preprocess_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#','', tweet)
    tweet = re.sub(r"[^a-zA-Z\s]", '', tweet)
    tweet = tweet.lower()
    word_tokens = word_tokenize(tweet)
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_tweet)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        processed_message = preprocess_tweet(message)
        message_tfidf = vectorizer.transform([processed_message])
        prediction = model.predict(message_tfidf)
        sentiment = 'positive' if prediction[0] == 1 else 'negative'
        return render_template('index.html', prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
