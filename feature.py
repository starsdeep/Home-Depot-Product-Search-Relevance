#encoding=utf8
import numpy as np
import re
import os
import pandas as pd
import time
from load_data import load_data
import hashlib
from utility import str_stem, len_of_str, num_whole_word, num_common_word, seg_words, num_size_word

total_train = 74067
total_test = 166693


def get_feature(config):
    feature = ' '.join(sorted(config['features']))
    feature_hash = hashlib.sha1(feature.encode('utf-8')).hexdigest()
    num_train = total_train if config['num_train']<0 else config['num_train']
    feature_filename = feature_hash + '_' + str(num_train)
    feature_path = './feature_cache/'
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    file_dict = {f.split('_')[0]: int(f.split('_')[1]) for f in os.listdir(feature_path) if os.path.isfile(os.path.join(feature_path, f))}

    if config['load_exist_feature'] and feature_hash in file_dict and num_train <= file_dict[feature_hash]:
        df = pd.read_csv(os.path.join(feature_path, feature_filename), encoding="ISO-8859-1", index_col=0)
        print("feature: " + feature + " already computed")
        print("load from " + feature_path + "/" + feature_filename)
    else:
        df, num_train, num_test = load_data(config['num_train'])
        print("feature not computed yet, start computing")
        start_time = time.time()
        df = build_feature(df, config['features'])
        print("--- Build Features: %s minutes ---" % round(((time.time() - start_time)/60),2))
        if config['save_feature']:
            df.to_csv(os.path.join(feature_path, feature_filename), encoding="utf8")
    print()
    return df[:num_train], df[num_train:]

def build_feature(df, features):
    for feature in features:
        feature_func = FeatureFuncDict[feature]
        df[feature] = feature_func(df)
    return df

def make_func(feature_name, func):
    return lambda df: df[feature_name].map(func)

def search_term_cut_(x):
    stop_w = ['for', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #, 'xbi']
    #x = (" ").join([z for z in x.split(" ") if z not in stop_w])

    has_core = False
    for s in x.split():
        is_number = re.match(r'^[0-9]+$', s)
        if (len(s)>2) and (not is_number) and (s not in stop_w):
            has_core = True
            break
    if not has_core:
        return ''
    else:
        return x

def search_term_cut_stem(df):
    return df['search_term'].map(lambda x:search_term_cut_(str_stem(x)))

def query_in_title(df):
    return df['tmp_compound_field'].map(lambda x: num_whole_word(x.split('\t')[0], x.split('\t')[1]))


def query_in_description(df):
    return df['tmp_compound_field'].map(lambda x: num_whole_word(x.split('\t')[0], x.split('\t')[2]))

def numsize_query_in_title(df):    
    return df['tmp_compound_field'].map(lambda x: num_size_word(x.split('\t')[0], x.split('\t')[1]))

def numsize_query_in_description(df):
    return df['tmp_compound_field'].map(lambda x: num_size_word(x.split('\t')[0], x.split('\t')[2]))

def query_last_word_in_title(df):
    return df['tmp_compound_field'].map(lambda x:num_common_word(x.split('\t')[0], x.split('\t')[1]))

def query_last_word_in_description(df):
    return df['tmp_compound_field'].map(lambda x:num_common_word(x.split('\t')[0], x.split('\t')[2]))

def word_in_title(df):
    return df['tmp_compound_field'].map(lambda x:num_common_word(x.split('\t')[0], x.split('\t')[1]))

def word_in_description(df):
    return df['tmp_compound_field'].map(lambda x:num_common_word(x.split('\t')[0], x.split('\t')[2]))

def word_in_brand(df):
    return df['tmp_compound_field'].map(lambda x:num_common_word(x.split('\t')[0], x.split('\t')[3]))

def search_term_fuzzy_match(df):
    return df['tmp_compound_field'].map(lambda x: seg_words(x.split('\t')[0], x.split('\t')[1]))


def count_er_word_in_(x):
    """
    function to count word which end with er x

    Example:

    >>> count_er_word_in_("col1 qfeqer 12 er adfer a")
    3
    >>> count_er_word_in_("col2 afdas 12")
    0
    """
    count = 0
    for word in x.split():
        if word[-2:] == "er":
            count += 1
    return count


def count_er_word(x, column):
    """
    function to count word which end with er,in column of x data frame

    Example:

    >>> import pandas as pd
    >>> ts1 = ("132er", "123123af", "13213afaer", "er", "12")
    >>> ts2 = ("132", "123123", "13213", "123", "123")
    >>> d = {'col1': ts1, 'col2': ts2}
    >>> df = pd.DataFrame(data=d)
    >>> count_er_word(df, "col1")
    0    1
    1    0
    2    1
    3    1
    4    0
    Name: col1, dtype: int64
    >>> count_er_word(df, "col2")
    0    0
    1    0
    2    0
    3    0
    4    0
    Name: col2, dtype: int64
    """
    return x[column].map(count_er_word_in_)


def word_with_er_count_in_query(df):
    return df['tmp_compound_field'].map(lambda x: count_er_word_in_(x.split('\t')[0]))


def word_with_er_count_in_title(df):
    return df['tmp_compound_field'].map(lambda x: count_er_word_in_(x.split('\t')[1]))


def find_er_position(query, title):
    position = -1
    first_er_in_title = ""
    for word in query.split():
        if word[-2:] == "er":
            first_er_in_title = word[-2:]
            break
    if first_er_in_title == "":
        return position

    for index, word in enumerate(title.split()):
        if word == first_er_in_title:
            position = index
            break
    return position


def first_er_in_query_occur_position_in_title(df):
    return df['tmp_compound_field'].map(lambda x: find_er_position(x.split("\t")[0], x.split("\t")[1]))



"""
现在这里有个做的不太好的地方，就是对column的处理采取map的形式，因此对某些feature的计算，如果需要2列以上，我们需要先把需要的那几列合起来，就是tmp_compound_column
以后再查查pandas的文档，看有没什么更好的方法
"""

FeatureFuncDict = {
    'search_term': search_term_cut_stem,
    'title': lambda df: df['product_title'].map(str_stem),
    'description': lambda df:df['product_description'].map(str_stem),
    'brand': lambda df:df['brand'].map(str_stem),
    'len_of_query': lambda df: df['search_term'].map(len_of_str).astype(np.int64),
    'len_of_title': lambda df: df['title'].map(len_of_str).astype(np.int64),
    'len_of_description': lambda df: df['description'].map(len_of_str).astype(np.int64),
    'len_of_brand': lambda df: df['brand'].map(len_of_str).astype(np.int64),

    'tmp_compound_field': lambda df:df['search_term'] + '\t' + df['title'] + '\t' + df['description'] + '\t' + df['brand'], # in order to using map, we need to make several fields into one
    'query_in_title': query_in_title,
    'query_in_description': query_in_description,
    'numsize_query_in_title': numsize_query_in_title,
    'numsize_query_in_description': numsize_query_in_description,
    'query_last_word_in_title': query_last_word_in_title,
    'query_last_word_in_description': query_last_word_in_description,
    'word_in_title': word_in_title,
    'word_in_description': word_in_description,
    'word_in_brand': word_in_brand,
    'ratio_title': lambda df :df['word_in_title'] / (df['len_of_query']+1),
    'ratio_description': lambda df :df['word_in_description'] / (df['len_of_query']+1),
    'ratio_brand': lambda df :df['word_in_brand'] / (df['len_of_query']+1),

    'search_term_fuzzy_match': search_term_fuzzy_match,
    'len_of_search_term_fuzzy_match': lambda df: df['search_term_fuzzy_match'].map(len_of_str).astype(np.int64),

    'word_with_er_count_in_query': word_with_er_count_in_query,
    'word_with_er_count_in_title': word_with_er_count_in_title,
    'first_er_in_query_occur_position_in_title': first_er_in_query_occur_position_in_title,

    'title_query_BM25': lambda df: df['title_query_BM25'] ,
    'description_query_BM25': lambda df: df['description_query_BM25']

}


if __name__=='__main__':
    import doctest
    doctest.testmod()
