import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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


def ratings_binarization(num):
    if num > 3:
        return 1
    elif num < 3:
        return -1
    else:
        return 0


# This function is used to fit different type of classifiers on featureset
# and calculate accuracy.
def sentiment_accuracy(features, ratings):
    actual_sentiment = list(map(ratings_binarization, ratings))
    train_x, test_x, train_y, test_y = train_test_split(
        features, actual_sentiment, test_size=0.2, random_state=42)
    clf1 = SVC()
    clf2 = GradientBoostingClassifier()
    clf3 = MultinomialNB()
    clf4 = DecisionTreeClassifier()
    labels = ['SVCClassifier', 'Xboost classfier', 'MultinomialNB', 'Decision tree classifier']
    classifiers = [clf1, clf2, clf3, clf4]
    for label, clf in zip(labels, classifiers):
        clf.fit(train_x, train_y)
        pred_y = clf.predict(test_x)
        print('{} accuracy'.format(label), 100 * accuracy_score(test_y, pred_y), '%')


def sentiment_analysis(reviews, ratings):
    reviews_text = []
    for _ in reviews:
        reviews_text.append(' '.join(_))
    bow_feature_set = bow(reviews_text, ngram=1).toarray()
    # tfidf_feature_set = tfidf(reviews_text,ngram=2).toarray()
    # np.savetxt("bow_unigram.csv",bow_feature_set,delimiter=',')
    sentiment_accuracy(bow_feature_set, ratings)
    # sentiment_accuracy(tfidf_feature_set,ratings)
    return


if __name__ == '__main__':
    with open("docs_preprocessed.pkl", "rb") as f:
        obj = pickle.load(f)
    reviews = obj[0][:500]
    ratings = obj[1][:500]
    sentiment_analysis(reviews
                       , ratings)
