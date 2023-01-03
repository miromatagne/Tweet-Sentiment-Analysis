from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

tweet = "What a great transfer! Sure you'll score plenty of goals 😉"

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

labels = ['Negative', 'Neutral', 'Positive']

encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')

output = model(**encoded_tweet)

scores = output[0][0].detach().numpy()
scores = softmax(scores)

maximal_score = max(scores)
most_probable_label = labels[np.argmax(scores)]

print(most_probable_label)
