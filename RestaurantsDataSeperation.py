import pickle

from pymongo import MongoClient

from topic_modeling import preprocess_data

client = MongoClient()
db = client.project


# This function is used to seperate the restaurant type business from other businesses
# in database As we working with restaurants data.
# This fuction return list of businessids which are restaurants type.
def seperate_restaurants_ids():
    cursor = db.business.find()
    resturants_business = []
    for row in cursor:
        categories = row['categories']
        # print(categories,type(categories))
        if 'Restaurants' in categories:
            resturants_business.append(row['business_id'])
    return resturants_business


# This function is used to seperate the restaurant reviews from other
# reviews in database.
def seperate_restaurants_reviews(noofdocs):
    restaurants_id = seperate_restaurants_ids()
    cursor = db.review.find()
    # print(cursor.count())
    docs = []
    ratings = []
    business_id = []
    cnt = 0
    for row in cursor:
        if row['business_id'] in restaurants_id:
            if cnt <= noofdocs:
                docs.append(row['text'])
                ratings.append(row['stars'])
                business_id.append(row['business_id'])
                cnt = cnt + 1
                if cnt % 10000 == 0:
                    print("Reviews completed {}".format(cnt))
            else:
                break
    save_object = (docs, ratings, business_id)
    with open("docs.pkl", "wb") as f:
        pickle.dump(save_object, f)


def generate_file(noofreviews):
    seperate_restaurants_reviews(noofreviews)
    with open("docs.pkl", "rb") as f:
        data = pickle.load(f)
    reviews = data[0][:noofreviews]
    ratings = data[1][:noofreviews]
    business_ids = data[2][:noofreviews]
    preprocess_data(reviews)
    with open("preprocessed_reviews.pkl", "rb") as f:
        preprocessed_reviews = pickle.load(f)
    preprocessed_reviews = preprocessed_reviews[:noofreviews]
    save_ob = (preprocessed_reviews, ratings, business_ids)
    with open("docs_preprocessed.pkl", "wb") as f:
        pickle.dump(save_ob, f)


if __name__ == '__main__':
    generate_file(1000000)
