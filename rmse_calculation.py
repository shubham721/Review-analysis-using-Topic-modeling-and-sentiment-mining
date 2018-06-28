import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# given a list of reviews, It converts them to bow with feature termfrequency.
def bow(reviews, ngram):
    vectorizer = CountVectorizer(ngram_range=(ngram, ngram), max_features=500)
    bow_features = vectorizer.fit_transform(reviews)
    # print(vectorizer.vocabulary_)
    # print(bow_features.shape)
    return bow_features


# given a list of reviews, It converts them to bow with feature tfidf.
def tfidf(reviews, ngram):
    vectorizer = TfidfVectorizer(ngram_range=(ngram, ngram), max_features=500)
    tfidf = vectorizer.fit_transform(reviews)
    # print(tfidf.shape)
    return tfidf


def calculation_rmse(pred_y, true_y):
    return np.sqrt(((pred_y - true_y) ** 2).mean())


# This function is used to fit the linear regression model on features then
# predict the rating on testset and calculate root mean square error.
def linear_model(features, labels):
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    return calculation_rmse(pred_y, test_y)


def ratings_binarization(num):
    if num > 3:
        return 1
    elif num < 3:
        return -1
    else:
        return 0


def rmse(reviews, ratings):
    reviews_text = []
    for _ in reviews:
        reviews_text.append(' '.join(_))
    # bow_feature_set=bow(reviews_text,ngram=2)
    bow_feature_set = bow(reviews_text, ngram=1)
    # tfidf_feature_set = tfidf(reviews_text,ngram=2)
    print("RMSE error: ", linear_model(bow_feature_set, ratings))
    # print(linear_model(tfidf_feature_set, ratings))
    return


if __name__ == '__main__':
    with open("docs_preprocessed.pkl", "rb") as f:
        obj = pickle.load(f)
    reviews = obj[0][:500]
    ratings = obj[1][:500]

    rmse(reviews, ratings)

# 50,000 reviews with 500 top features(based on term frequency)
# SVCClassifier accuracy 76.4 %
# MultinomialNB accuracy 75.4 %
# Xboost classfierNearestNeighbourClassifier accuracy 75.08 %
# DecisionTreeClassifier accuracy 66.85 %


# 100000 reviews with 500 bigram+tf
# SVC 67.88%
# xboost 69.32
# MNB 70.49
# DCT 61.82


# 100000 reviews with 500 unigram+tfidf
# svc 70.145%
# xboost 76.559%
# mnb 75%
# dct 61.5%
