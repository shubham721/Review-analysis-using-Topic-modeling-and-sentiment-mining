import math
import re
from sklearn.metrics.classification import accuracy_score
import pickle

filenameAFINN = 'AFINN/AFINN-111.txt'
afinn_in = [ws.strip().split('\t') for ws in open(filenameAFINN)]
afinn = []
for temp in afinn_in:
    if len(temp) == 2:
        afinn.append(temp)
afinn = dict(afinn)
# Word splitter pattern
pattern_split = re.compile(r"\W+")


def sentiment_afinn_accuracy(preprocessed_reviews, ratings):
    print(preprocessed_reviews[:10])
    pred_sentiment = list(map(sentiment, preprocessed_reviews))
    print(pred_sentiment)

    def ratings_binarization(num):
        if num >= 3:
            return 1
        else:
            return 0

    rating_sentiment = list(map(ratings_binarization, ratings))
    print(rating_sentiment)
    print(accuracy_score(rating_sentiment, pred_sentiment))


def sentiment(words):
    # words = pattern_split.split(text.lower())
    sentiments = list(map(lambda word: int(afinn.get(word, 0)), words))
    # print(sentiments)
    if sentiments:

        sentiment = float(sum(sentiments)) / math.sqrt(len(sentiments))

    else:
        sentiment = 0
    if sentiment >= 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    with open("docs_preprocessed.pkl", "rb") as f:
        obj = pickle.load(f)
    reviews = obj[0][:500]
    ratings = obj[1][:500]
    sentiment_afinn_accuracy(reviews, ratings)
