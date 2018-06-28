Topic Modeling on reviews and Sentiment mining.
---------------------------------------------------------------

Dataset Used:
We have used the dataset provided by Yelp for exploratory, topics clustering, sentiment analysis and
prediction of ratings. The dataset contains records from US, UK, Canada and Germany. It contains
information on businesses, business attributes, check-in sets, tips and text reviews.
The dataset consists of five JSON files, namely: business, review, user, check-in, and tip JSON objects. We are
working specifically on Restaurants business data, so We have wrote a script RestaurantDataSeperate.py for seperating
restaurant data, Preprocess it and then we have pickle data into docs_preprocessed.pkl.

In this project, we have imported review.json, business.json files into mongodb database using mongodb shell. you
need to install mongodb before running mongodb shell. Then you can import json file using shell command like this.
command : mongoimport --db users --collection contacts --file contacts.json
Above command imports the JSON data from the contacts.json file into the collection contacts in the users database.

------------------------------------------------------------------------
Dataset Link:
https://www.yelp.com/dataset/download
------------------------------------------------------------------------


Problem:
----------------------------------------------------------------
This project aims to do following on reviews
1. Topic modeling using latent Dirichlet algorithm (LDA)
2. Sentiment Mining using AFINN and different classifiers like Xgboost, svm, Naive-bayes etc.
3. Prediction of ratings to check reviews if they are biased and calculate Root Mean Square Error.

Requirements:
----------------------------------------------------------------
1)Python3 (Python3 with Anaconda recommended)
2)gensim (This package is used for lda algo)
3)pymongo
4)Numpy, scikit-learn
5)plotly (This tool is used for plotting the graphs.)
6)pickle

Modules Information:
---------------------------------------------------------------------

RestaurantDataSeperation.py: This module is used to seperate the Restaurants Category reviews from others.

topic_modeling.py : This module is used for topic modeling on reviews.

sentiment_afinn.py: This module is used to do sentiment analysis of reviews by using AFINN.

sentimentanalysis_classifiers.py: This  module is used to fit different type of classifier like svc,naivebayes, xboost etc. on
featureset and predict the sentiment.


rmse_calculation.py: This module is used to predict the ratings on reviews by fitting a linear regression model
on reviews and then calculate rmse on testset.


stopwords.txt: This file contain stopwords for english.

model.html : It contains the approach used for the project.

docs_preprocessed.pkl --> This is a file which contain a list of tuple.Each tuple contain a preprocessed
review, its rating and corresponding business id through which review is generates. These all are restaurants
reviews which are extracted from database and save in to this file so you don't need to retrieve database
again and again.


How To Run.
---------------------------------------------------------------------------------------------

1)Run 'python RestaurantsDataSeperation.py' to seperate the restaurants data and generate
    docs_preprocessed.pkl, It is used by all other modules for reviews.
2) Run 'python topic_modeling.py' to do the topic analysis on reviews and extract the topics.
3) Run 'python sentiment_afinn.py' to do the sentiment analysis on reviews using AFINN.
4) Run 'python sentimentanalysis_classifiers.py' to do the sentiment analysis on reviews using variety
    of classifiers like xgboost,svm, Naive-bayes etc.
5) Run 'python rmse_calculation.py' to predict ratings on reviews by fitting a linear regression model and check
    How much reviews are biased as a measure of Root Mean Square Error(RMSE).




