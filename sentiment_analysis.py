from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import tweepy


labels = ['Negative', 'Neutral', 'Positive']


def get_sentiment(tweet):
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'

        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)

    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')

    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    most_probable_label = labels[np.argmax(scores)]

    return most_probable_label


API_KEY = "FlSBNE9CT4BxneLq2XjTldPZb"
API_KEY_SECRET = "yhbtcBlTnOxNPFE9GUWbGlwZV9CYaF2j1Cak212dvjj6BgPLai"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAMwAlAEAAAAAbAkPEPiFacKC26sNUpPhSN7SBcE%3Dew9hLSdJTR9TuxmL15IRJWIVx7LcIdBLM593QCDyNN5IgLcL7b"
ACCESS_TOKEN = "1610335659910119425-RbXzErygpkolhRIcdKRPz3wAx2HRrY"
ACCESS_TOKEN_SECRET = "kY22YcW29DIIteugQpwA4cH5p31AjeLQx9QDntLA7HtwB"

client = tweepy.Client(bearer_token=BEARER_TOKEN, consumer_key=API_KEY, consumer_secret=API_KEY_SECRET,
                       access_token=ACCESS_TOKEN, access_token_secret=ACCESS_TOKEN_SECRET, return_type=dict)


keywords = "Messi -is:retweet lang:en"
date_since = "2023-01-02"

fetched_tweets = client.search_recent_tweets(query=keywords, max_results=10)
sentiment_count = {i: 0 for i in labels}

for tweet in fetched_tweets['data']:
    print(tweet['text'])
    sentiment = get_sentiment(tweet['text'])
    print(sentiment)
    sentiment_count[sentiment] += 1

print(sentiment_count)
