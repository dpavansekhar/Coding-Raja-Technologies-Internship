import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

# Streamlit app
st.title('Sentiment Analysis')

st.write("""
Enter a tweet below and click 'Analyze Sentiment' to see the predicted sentiment (positive or negative).
""")

tweet = st.text_area("Enter your tweet here...")

if st.button('Analyze Sentiment'):
    if tweet:
        processed_tweet = preprocess_tweet(tweet)
        tweet_tfidf = vectorizer.transform([processed_tweet])
        prediction = model.predict(tweet_tfidf)
        sentiment = 'positive' if prediction[0] == 1 else 'negative'
        st.write(f'**Predicted Sentiment:** {sentiment}')
    else:
        st.write("Please enter a tweet to analyze.")

