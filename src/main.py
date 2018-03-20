import gzip
from nltk.tokenize import word_tokenize
import json
from random import shuffle
from math import log, fabs
from nltk.corpus import stopwords


ENGLISH_STOP_WORDS = stopwords.words('english')


def load_data():
    positive = open('../resource/large_data_2000')
    positive_lines = positive.readlines()
    positive.close()
    return positive_lines


def get_label(rating):
    return 'positive' if int(rating) > 3 else 'negative'


def find_important_words_in_sentence(sentence):
    tokens = list(set(word_tokenize(sentence.replace('.', ' ').lower())))
    return [word for word in tokens if len(word) >= 2 and word not in ENGLISH_STOP_WORDS]


def extract_text_and_label(review):
    review_features = json.loads(review)
    review_text = find_important_words_in_sentence(review_features['reviewText'])
    label = get_label(review_features['overall'])
    return review_text, label


def extract_text_and_rating(reviews):
    review_data = []
    for review in reviews:
        review_text, review_rating = extract_text_and_label(review)
        review_data.append((review_text, review_rating))

    return review_data


def load_model(model_file):
    with open(model_file, 'r') as model_file:
        model = json.loads(model_file.read())
        return model['word_probabilities'], model['class_probability']


def classify(review, word_prob, class_prob):
    prob_positive = 0
    prob_negative = 0
    prob_words_p = 0.0
    prob_words_n = 0.0
    for w in review:
        if w in word_prob and word_prob[w]['positive'] != 0:
            prob_positive += log(word_prob[w]['positive'])
            prob_words_p += log(word_prob[w]['all'])

        if w in word_prob and word_prob[w]['negative'] != 0:
            prob_negative += log(word_prob[w]['negative'])
            prob_words_n += log(word_prob[w]['all'])

    prob_positive = prob_positive + log(class_prob['positive']) - prob_words_p
    prob_negative = prob_negative + log(class_prob['negative']) - prob_words_n

    return prob_positive, prob_negative, 'positive' if prob_positive > prob_negative else 'negative'


if __name__ == '__main__':
    word_prob, class_prob = load_model('../resource/model-1521523880.34')
    reviews = load_data()
    shuffle(reviews)
    test_data = extract_text_and_rating(reviews)
    result_stat = {'positive': {'positive': 0, 'negative': 0}, 'negative': {'positive': 0, 'negative': 0}}
    for item, label in test_data:
        p, n, r = classify(item, word_prob, class_prob)
        result_stat[label][r] += 1

    print result_stat
