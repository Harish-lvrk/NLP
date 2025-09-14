import pickle
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from difflib import SequenceMatcher
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

# --- 1. LOAD ALL SAVED ARTIFACTS AT STARTUP ---
try:
    model = pickle.load(open('model.pkl', 'rb'))
    cv = pickle.load(open('cv.pkl', 'rb'))
    stop_words = pickle.load(open('stopwords.pkl', 'rb'))
except FileNotFoundError:
    model, cv, stop_words = None, None, None
    print("Warning: One or more .pkl files not found. The app will not work without model.pkl, cv.pkl, and stopwords.pkl.")


# --- 2. RECREATE THE EXACT SAME PREPROCESSING AND FEATURE ENGINEERING FUNCTIONS ---

def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent'); q = q.replace('$', ' dollar '); q = q.replace('₹', ' rupee '); q = q.replace('€', ' euro '); q = q.replace('@', ' at ')
    q = q.replace('[math]', '')
    q = q.replace(',000,000,000 ', 'b '); q = q.replace(',000,000 ', 'm '); q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q); q = re.sub(r'([0-9]+)000000', r'\1m', q); q = re.sub(r'([0-9]+)000', r'\1k', q)
    contractions = {"ain't": "am not", "aren't": "are not","can't": "can not", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not", "must've": "must have", "mustn't": "must not", "needn't": "need not", "o'clock": "of the clock", "oughtn't": "ought not", "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have", "shouldn't": "should not", "so've": "so have", "so's": "so is", "that'd": "that would", "that's": "that is", "there'd": "there would", "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have", "where'd": "where did", "where's": "where is", "who'll": "who will", "who's": "who is", "who've": "who have", "why's": "why is", "won't": "will not", "would've": "would have", "wouldn't": "would not", "y'all": "you all", "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have"}
    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)
    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have"); q = q.replace("n't", " not"); q = q.replace("'re", " are"); q = q.replace("'ll", " will")
    # Correctly parsing HTML
    q = BeautifulSoup(q, "html.parser").get_text()
    # Removing all non-alphanumeric characters
    q = re.sub(r'[^\w\s]', '', q)
    return q

def get_all_features(row):
    q1 = row['question1']
    q2 = row['question2']
    features = {}
    features['q1_len_char'] = len(q1); features['q2_len_char'] = len(q2)
    q1_tokens = q1.split(); q2_tokens = q2.split()
    features['q1_len_words'] = len(q1_tokens); features['q2_len_words'] = len(q2_tokens)
    SAFE_DIV = 0.0001
    common_word_set = set(q1_tokens) & set(q2_tokens)
    common_word_count = len(common_word_set)
    q1_stopwords = [word for word in q1_tokens if word in stop_words]
    q2_stopwords = [word for word in q2_tokens if word in stop_words]
    common_stop_count = len(set(q1_stopwords) & set(q2_stopwords))
    features['cwc_min'] = common_word_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    features['cwc_max'] = common_word_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    features['csc_min'] = common_stop_count / (min(len(q1_stopwords), len(q2_stopwords)) + SAFE_DIV)
    features['csc_max'] = common_stop_count / (max(len(q1_stopwords), len(q2_stopwords)) + SAFE_DIV)
    features['ctc_min'] = common_word_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    features['ctc_max'] = common_word_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    features['last_word_eq'] = 1 if len(q1_tokens) > 0 and len(q2_tokens) > 0 and q1_tokens[-1] == q2_tokens[-1] else 0
    features['first_word_eq'] = 1 if len(q1_tokens) > 0 and len(q2_tokens) > 0 and q1_tokens[0] == q2_tokens[0] else 0
    features['mean_len'] = (len(q1_tokens) + len(q2_tokens)) / 2
    features['abs_len_diff'] = abs(len(q1_tokens) - len(q2_tokens))
    match = SequenceMatcher(None, q1, q2).find_longest_match(0, len(q1), 0, len(q2))
    features['longest_substr_ratio'] = match.size / (min(len(q1), len(q2)) + SAFE_DIV)
    features['fuzz_ratio'] = fuzz.ratio(q1, q2)
    features['fuzz_partial_ratio'] = fuzz.partial_ratio(q1, q2)
    features['token_sort_ratio'] = fuzz.token_sort_ratio(q1, q2)
    features['token_set_ratio'] = fuzz.token_set_ratio(q1, q2)
    return pd.Series(features)

def query_point_creator(q1, q2):
    if not all([model, cv, stop_words]):
        return None

    new_df = pd.DataFrame({'question1': [q1], 'question2': [q2]})
    new_df['question1'] = new_df['question1'].apply(preprocess)
    new_df['question2'] = new_df['question2'].apply(preprocess)
    feature_df = new_df.apply(get_all_features, axis=1)

    questions = list(new_df['question1']) + list(new_df['question2'])
    questions_bow = cv.transform(questions)

    q1_arr_sparse = questions_bow[0]
    q2_arr_sparse = questions_bow[1]
    bow_sparse = hstack((q1_arr_sparse, q2_arr_sparse)).tocsr()

    engineered_features_sparse = csr_matrix(feature_df.values)
    final_sparse_matrix = hstack((engineered_features_sparse, bow_sparse)).tocsr()

    return final_sparse_matrix

