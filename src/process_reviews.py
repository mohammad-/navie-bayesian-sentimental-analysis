from nltk.tokenize import word_tokenize
import json
from nltk.corpus import stopwords
import enchant
import time

ENGLISH_STOP_WORDS = stopwords.words('english')
ENGLISH_WORDS = enchant.Dict('en_US')


def find_important_words_in_sentence(sentence):
    tokens = list(set(word_tokenize(sentence.replace('.', ' ').lower())))
    return [word for word in tokens if len(word) >= 2 and ENGLISH_WORDS.check(word) and word not in ENGLISH_STOP_WORDS]


def get_label(rating):
    return 'positive' if int(rating) > 3 else 'negative'


def extract_text_and_label(review):
    review_features = json.loads(review)
    review_text = find_important_words_in_sentence(review_features['reviewText'])
    label = get_label(review_features['overall'])
    return review_text, label


def calculate_probability(feature_set):
    feature_prob = {}
    # This dictionary holds total number of positive and negative reviews
    review_by_class = {'positive': 0., 'negative': 0.}
    for review, label in feature_set:
        review_by_class[label] += 1
        for word in review:
            if word not in feature_prob.keys():
                # Initialize an dictionary to store count of each word.
                # positive: how many times word appears in positive review.
                # negative: how many times word applears in negative review.
                # all: how many times word appear in general.
                # We count a word only once in a review.
                feature_prob[word] = {'positive': 0., 'negative': 0., 'all': 0.}

            feature_prob[word][label] += 1
            feature_prob[word]['all'] += 1

    for f in feature_prob.keys():
        # P(Word given review is positive)
        feature_prob[f]['positive'] = feature_prob[f]['positive'] / review_by_class['positive']
        # P(Word given negative is positive)
        feature_prob[f]['negative'] = feature_prob[f]['negative'] / review_by_class['negative']
        # P(Word in dataset)
        feature_prob[f]['all'] = feature_prob[f]['all'] / (review_by_class['positive'] + review_by_class['negative'])

    return feature_prob


if __name__ == '__main__':
    data_set = open('../resource/train_data')
    features = [extract_text_and_label(review) for review in data_set]
    model_data = calculate_probability(features)
    filename = '../resource/model-%s' % str(time.time())
    review_positive_prob = 0.7161666667
    review_negative_prov = 1.0 - review_positive_prob
    model = {'word_probabilities': model_data,
             'class_probability':
                 {'positive': review_positive_prob,
                  'negative': review_negative_prov}}
    model_text = json.dumps(model, indent=4)
    with open(filename, 'w') as model_file:
        model_file.write(model_text)
