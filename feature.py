#encoding=utf8
import numpy as np
import re
import os
import pandas as pd
import time
from load_data import load_data
import hashlib
from utility import *
from SpellCorrect import *
from collections import OrderedDict
from nltk import pos_tag

total_train = 74067
total_test = 166693

feature_path = './output/features/'


# def load_feature(config):
#     df = pd.read_csv(config["load_feature_from"], encoding="ISO-8859-1", index_col=0)
#     num_train = total_train if config['num_train'] < 0 else config['num_train']
#     return df[:num_train], df[total_test * -1:]
#
#
#
#     names = [os.path.splitext(filename)[0] for filename in filenames]
#
#     return names, paths



def load_feature(features):
    """
    read features from existing files, and then concat those features into a big frame using pd.concat,
    concat a list of frames can improve efficiency see http://pandas.pydata.org/pandas-docs/stable/merging.html
    for more info.
    :param features: features to load
    :return: df contains the features
    """
    frames = []
    files = [os.path.join(feature_path, feature + '.csv') for feature in features]
    frames = [pd.read_csv(file, encoding="ISO-8859-1", index_col=0) for file in files]
    df = pd.concat(frames, axis=1)

    # read_csv() fills empty string with nan, which will cause a problem in build_feature, so we need to replace nan to '', see https://github.com/pydata/pandas/issues/10205 for detail
    df.fillna('', inplace=True)
    return df



def write_feature(df, features):
    for feature in features:
        tmp_df = df[[feature]]
        tmp_df.to_csv(os.path.join(feature_path, feature + '.csv'), encoding="utf8")
    return


def get_feature(config):
    all_exist_features = set([os.path.splitext(f)[0] for f in os.listdir(feature_path) if os.path.isfile(os.path.join(feature_path,f)) and f.endswith('.csv')])
    total_features = set(config['features'])
    exist_features = total_features & all_exist_features
    recompute_features = set(config['recompute_features']) if 'recompute_features' in config else set()
    new_features = total_features - exist_features

    to_load_features = exist_features - recompute_features
    to_compute_features = recompute_features | new_features

    df_basic, num_train, num_test = load_data(config['num_train'])
    print("feature already exists, loading: \n" + ' '.join(to_load_features))
    if to_load_features:
        df_all = load_feature(to_load_features)
        print("length of loaded datafame %d" % df_all.shape[0])
        df_train = df_all[:num_train]
        df_test = df_all[-num_test:]
        df = pd.concat((df_train, df_test), axis=0, ignore_index=True)
        print("length of datafame after trimed with num_train %d" % df.shape[0])
        for column in df_basic.columns.values:
            if column not in df:
                df[column] = df_basic[column].copy()
    else:
        df = df_basic
    print("loading done")
    print("start computing feature: " + ' '.join(to_compute_features))
    df = build_feature(df, to_compute_features)
    write_feature(df, to_compute_features)


    return df[:num_train], df[num_train:]


# def get_feature(config):
#     feature = ' '.join(sorted(config['features']))
#     feature_hash = hashlib.sha1(feature.encode('utf-8')).hexdigest()
#     num_train = total_train if config['num_train'] < 0 else config['num_train']
#     feature_filename = feature_hash + '_' + str(num_train)
#     feature_path = './feature_cache/'
#     if not os.path.exists(feature_path):
#         os.makedirs(feature_path)
#
#     file_dict = dict()
#     for f in os.listdir(feature_path):
#         if os.path.isfile(os.path.join(feature_path, f)):
#             hash_value = f.split('_')[0]
#             num_value = int(f.split('_')[1])
#             if hash_value not in file_dict or num_value>file_dict[hash_value]:
#                 file_dict[hash_value] = num_value
#
#     if feature_hash in file_dict and num_train <= file_dict[feature_hash]:
#         df = pd.read_csv(os.path.join(feature_path, feature_filename), encoding="ISO-8859-1", index_col=0)
#         print("feature: " + feature + " already computed")
#         print("load from " + feature_path + "/" + feature_filename)
#     else:
#         df, num_train, num_test = load_data(config['num_train'])
#         print("feature not computed yet, start computing")
#         start_time = time.time()
#         df = build_feature(df, config['features'])
#         print("--- Build Features: %s minutes ---" % round(((time.time() - start_time)/60),2))
#         df.to_csv(os.path.join(feature_path, feature_filename), encoding="utf8")
#
#     return df[:num_train], df[num_train:]

def build_feature(df, features):
    # iterate features in order, use apply() to update in time
    if not features:
        return df
    for feature in list(FirstFeatureFuncDict.keys()):
        if feature in features:
            print('calculating feature: '+feature+' ...')
            feature_func = FirstFeatureFuncDict[feature]
            df[feature] = df.apply(feature_func, axis=1)

    # iterate features in order (iterrows cannot update in time)
    if set(features) & set(PostagFeatureFuncDict.keys()):
        print('calculating pos_tag features...')
        for index, row in df.iterrows():
            tags = {'search_term': pos_tag(row['search_term'].split()),
                    'main_title': pos_tag(row['main_title'].split()),
                    'title': pos_tag(row['title'].split())}
            # caution: takes a long time
            if ('noun_of_description' in features) or ('noun_match_description' in features):
                tags['description'] = pos_tag(row['description'].split())
            for feature in list(PostagFeatureFuncDict.keys()):
                if feature in features:
                    feature_func = PostagFeatureFuncDict[feature]
                    df.loc[index, feature] = feature_func(row, tags)
            if index%300==0:
                print(str(index)+' rows calculated...')

    # compute CategoricalNumsizeFuncDict
    if set(features) & set(NumsizeFuncDict.keys()):
        print('calculating cateforical numsize features...')
        for index, row in df.iterrows():
            help_dict = {
                'numsize_of_query': numsize_of_str(row['search_term']),
                'numsize_of_title': numsize_of_str(row['title']),
                'numsize_of_main_title': numsize_of_str(row['main_title']),
                'numsize_of_description': numsize_of_str(row['description']),
            }
            for feature in list(NumsizeFuncDict.keys()):
                if feature in features:
                    feature_func = NumsizeFuncDict[feature]
                    df.loc[index, feature] = feature_func(row, help_dict)
            if index%500==0:
                print(str(index)+' rows calculated...')

    # iterate features in order, use apply() to update in time
    for feature in list(LastFeatureFuncDict.keys()):
        if feature in features:
            print('calculating feature: '+feature+' ...')
            feature_func = LastFeatureFuncDict[feature]
            df[feature] = df.apply(feature_func, axis=1)
    return df

chkr = SpellCheckGoogleOffline()
def search_term_clean(query):
    query = chkr.spell_correct(query)
    query = str_stem(query)
    query = query if str_is_meaningful(query) else ''
    query = str_remove_stopwords(query)
    return query

def last_word_in_title(s, t):
    """
        How many times last word of s occurs in t 
    """
    words = s.split()
    if len(words)==0:
        return 0
    return num_common_word(words[-1], t)

# Following features in dicts will be calculated from top to bottom

# Features can be calculated by raw input in order
FirstFeatureFuncDict = OrderedDict([
    ('origin_search_term', lambda row: row['search_term']),
    ('ori_stem_search_term', lambda row: str_stem(row['search_term'])),
    ('search_term', lambda row: search_term_clean(row['search_term'])),
    ('typeid', lambda row: str_stem(typeid_extract(row['product_title']))),
    ('main_title', lambda row: str_stem(main_title_extract(row['product_title']))),
    ('title', lambda row: str_stem(row['product_title'])),
    ('description', lambda row: str_stem(row['product_description'])),
    ('brand', lambda row: str_stem(row['brand'])),

    ('ori_query_in_title', lambda row: num_whole_word(row['ori_stem_search_term'], row['title'])),
    ('query_in_main_title', lambda row: num_whole_word(row['search_term'], row['main_title'])),
    ('query_in_title', lambda row: num_whole_word(row['search_term'], row['title'])),
    ('query_in_description', lambda row: num_whole_word(row['search_term'], row['description'])),   
    ('query_last_word_in_main_title', lambda row: last_word_in_title(row['search_term'], row['main_title'])),
    ('query_last_word_in_title', lambda row: last_word_in_title(row['search_term'], row['title'])),
    ('query_last_word_in_description', lambda row: last_word_in_title(row['search_term'], row['description'])),
    ('word_in_main_title', lambda row: num_common_word(row['search_term'], row['main_title'])),
    ('word_in_main_title_exact', lambda row: num_common_word(row['search_term'], row['main_title'], exact_matching=True)),
    ('word_in_main_title_ordered', lambda row: num_common_word_ordered(row['search_term'], row['main_title'])),
    ('word_in_title', lambda row: num_common_word(row['search_term'], row['title'])),

    ('word_in_title_exact', lambda row: num_common_word(row['search_term'], row['title'], exact_matching=True)),

    ('ori_word_in_title_ordered', lambda row: num_common_word_ordered(row['ori_stem_search_term'], row['title'])),
    ('word_in_title_ordered', lambda row: num_common_word_ordered(row['search_term'], row['title'])),
    ('word_in_description', lambda row: num_common_word(row['search_term'], row['description'])),
    ('word_in_description_exact', lambda row: num_common_word(row['search_term'], row['description'], exact_matching=True)),
    ('word_in_brand', lambda row: num_common_word(row['search_term'], row['brand'])),
    ('word_in_typeid', lambda row: num_common_word(row['search_term'], row['typeid'], exact_matching=False)),

    ('bigram_in_title', lambda row: num_common_word(row['search_term'], row['title'], ngram=2)),
    ('bigram_in_main_title', lambda row: num_common_word(row['search_term'], row['main_title'], ngram=2)),
    ('bigram_in_description', lambda row: num_common_word(row['search_term'], row['description'], ngram=2)),
    ('bigram_in_brand', lambda row: num_common_word(row['search_term'], row['brand'], ngram=2)),

    ('search_term_fuzzy_match', lambda row: seg_words(row['search_term'], row['title'])),
    ('len_of_search_term_fuzzy_match', lambda row: words_of_str(row['search_term_fuzzy_match'])),

    ('word_with_er_count_in_query', lambda row: count_er_word_in_(row['search_term'])),
    ('word_with_er_count_in_title', lambda row: count_er_word_in_(row['title'])),
    ('first_er_in_query_occur_position_in_title', lambda row: find_er_position(row['search_term'], row['title'])),

    ('len_of_query', lambda row: words_of_str(row['search_term'])),
    ('len_of_main_title', lambda row: words_of_str(row['main_title'])),
    ('len_of_title', lambda row: words_of_str(row['title'])),
    ('len_of_description', lambda row: words_of_str(row['description'])),
    ('len_of_brand', lambda row: words_of_str(row['brand'])),

    ('len_of_typeid', lambda row: words_of_str(row['typeid'])),

    ('chars_of_query', lambda row: len(row['search_term'])),

    ('ratio_main_title', lambda row :row['word_in_main_title'] / (row['len_of_query']+1.0)),
    ('ratio_title', lambda row :row['word_in_title'] / (row['len_of_query']+1.0)),
    ('ratio_main_title_exact', lambda row :row['word_in_main_title_exact'] / (row['len_of_query']+1.0)),
    ('ratio_title_exact', lambda row :row['word_in_title_exact'] / (row['len_of_query']+1.0)),
    ('ratio_main_title_ordered', lambda row :row['word_in_main_title_ordered'] / (row['len_of_query']+1.0)),
    ('ratio_title_ordered', lambda row :row['word_in_title_ordered'] / (row['len_of_query']+1.0)),
    ('ratio_description', lambda row :row['word_in_description'] / (row['len_of_query']+1.0)),
    ('ratio_description_exact', lambda row :row['word_in_description_exact'] / (row['len_of_query']+1.0)),
    ('ratio_brand', lambda row :row['word_in_brand'] / (row['len_of_query']+1.0)),

    ('ratio_bigram_title', lambda row: row['bigram_in_title'] / (row['len_of_query']+1.0)),
    ('ratio_bigram_main_title', lambda row: row['bigram_in_main_title'] / (row['len_of_query']+1.0)),
    ('ratio_bigram_description', lambda row: row['bigram_in_description'] / (row['len_of_query']+1.0)),
    ('ratio_bigram_brand', lambda row: row['bigram_in_brand'] / (row['len_of_query']+1.0)),


    ('title_query_BM25', lambda row: row['title_query_BM25']),
    ('description_query_BM25', lambda row: row['description_query_BM25'])
])

# Features dependending on pos_tag dict
PostagFeatureFuncDict = OrderedDict([
    ('noun_of_query', lambda row, tags: noun_of_str(tags['search_term'])),
    ('noun_of_title', lambda row, tags: noun_of_str(tags['title'])),
    ('noun_of_main_title', lambda row, tags: noun_of_str(tags['main_title'])),
    ('noun_of_description', lambda row, tags: noun_of_str(tags['description'])),
    ('noun_match_main_title', lambda row, tags: num_common_noun(row['search_term'], tags['main_title'])),
    ('noun_match_title', lambda row, tags: num_common_noun(row['search_term'], tags['title'])),
    ('noun_match_main_title_ordered', lambda row, tags: num_common_noun_ordered(row['search_term'], tags['main_title'])),
    ('noun_match_title_ordered', lambda row, tags: num_common_noun_ordered(row['search_term'], tags['title'])),
    ('noun_match_description', lambda row, tags: num_common_noun(row['search_term'], tags['description'])),
    ('match_last_noun_main', lambda row, tags: match_last_k_noun(row['search_term'], tags['main_title'], 1)),
    ('match_last_2_noun_main', lambda row, tags: match_last_k_noun(row['search_term'], tags['main_title'], 2)),
    ('match_last_3_noun_main', lambda row, tags: match_last_k_noun(row['search_term'], tags['main_title'], 3)),
    ('match_last_5_noun_main', lambda row, tags: match_last_k_noun(row['search_term'], tags['main_title'], 5)) # average nouns in main of all data: 5.3338
])


# categorical feature for numsize, 用来表征query和title,description的匹配情况的feature,是binary feature
NumsizeFuncDict = OrderedDict([
    ('numsize_word_in_main_title', lambda row, help_dict: num_numsize_word(help_dict['numsize_of_query'], row['main_title'])),
    ('numsize_word_in_title', lambda row, help_dict: num_numsize_word(help_dict['numsize_of_query'], row['title'])),
    ('numsize_word_in_description', lambda row, help_dict: num_numsize_word(help_dict['numsize_of_query'], row['description'])),
    ('numsize_match_title', lambda row, help_dict: num_common_word(row['search_term'], " ".join(help_dict['numsize_of_title']), exact_matching=False)),
    ('numsize_match_description', lambda row, help_dict: num_common_word(row['search_term'], " ".join(help_dict['numsize_of_description']), exact_matching=False)),
    ('numsize_match_title_exact', lambda row, help_dict: num_common_word(row['search_term'], " ".join(help_dict['numsize_of_title']), exact_matching=True)),
    ('numsize_match_description_exact', lambda row, help_dict: num_common_word(row['search_term'], " ".join(help_dict['numsize_of_description']), exact_matching=True)),
    ('len_of_numsize_query', lambda row, help_dict: len(help_dict['numsize_of_query'])),
    ('len_of_numsize_main_title', lambda row, help_dict: len(help_dict['numsize_of_main_title'])),
    ('len_of_numsize_title', lambda row, help_dict: len(help_dict['numsize_of_title'])),
    ('len_of_numsize_description', lambda row, help_dict: len(help_dict['numsize_of_description'])),

    ('numsize_title_case1', lambda row, help_dict: len(help_dict['numsize_of_query'])==0 and len(help_dict['numsize_of_title'])==0),
    ('numsize_title_case2', lambda row, help_dict: len(help_dict['numsize_of_query'])==0 and len(help_dict['numsize_of_title'])>0),
    ('numsize_title_case3', lambda row, help_dict: len(help_dict['numsize_of_query'])>0 and len(help_dict['numsize_of_title'])==0),
    ('numsize_title_case4', lambda row, help_dict: len(help_dict['numsize_of_query'])>0 and len(help_dict['numsize_of_title'])>0 and len(set(help_dict['numsize_of_query']) & set(help_dict['numsize_of_title']))==0),
    ('numsize_title_case5', lambda row, help_dict: len(help_dict['numsize_of_query'])>0 and len(help_dict['numsize_of_title'])>0 and len(set(help_dict['numsize_of_query']) & set(help_dict['numsize_of_title']))>0),
    ('numsize_description_case1', lambda row, help_dict: len(help_dict['numsize_of_query'])==0 and len(help_dict['numsize_of_description'])==0),
    ('numsize_description_case2', lambda row, help_dict: len(help_dict['numsize_of_query'])==0 and len(help_dict['numsize_of_description'])>0),
    ('numsize_description_case3', lambda row, help_dict: len(help_dict['numsize_of_query'])>0 and len(help_dict['numsize_of_description'])==0),
    ('numsize_description_case4', lambda row, help_dict: len(help_dict['numsize_of_query'])>0 and len(help_dict['numsize_of_description'])>0 and len(set(help_dict['numsize_of_query']) & set(help_dict['numsize_of_description']))==0),
    ('numsize_description_case5', lambda row, help_dict: len(help_dict['numsize_of_query'])>0 and len(help_dict['numsize_of_description'])>0 and len(set(help_dict['numsize_of_query']) & set(help_dict['numsize_of_description']))>0),
])

# Statistical Features
LastFeatureFuncDict = OrderedDict([
    ('ratio_noun_match_title', lambda row: row['noun_match_title'] / (row['noun_of_query']+1)),
    ('ratio_noun_match_main_title', lambda row: row['noun_match_main_title'] / (row['noun_of_query']+1)),

    ('ratio_noun_match_title_ordered', lambda row: row['noun_match_title_ordered'] / (row['noun_of_query']+1)),
    ('ratio_noun_match_main_title_ordered', lambda row: row['noun_match_main_title_ordered'] / (row['noun_of_query']+1)),
    ('ratio_noun_match_description', lambda row: row['noun_match_description'] / (row['noun_of_query']+1)),

    ('ratio_numsize_main_title', lambda row :row['numsize_word_in_main_title'] / (row['len_of_numsize_query']+1)),
    ('ratio_numsize_title', lambda row :row['numsize_word_in_title'] / (row['len_of_numsize_query']+1)),
    ('ratio_numsize_description', lambda row :row['numsize_word_in_description'] / (row['len_of_numsize_query']+1)),
    ('ratio_numsize_match_title', lambda row :row['numsize_match_title'] / (row['len_of_numsize_query']+1)),
    ('ratio_numsize_match_description', lambda row :row['numsize_match_description'] / (row['len_of_numsize_query']+1)),
    ('ratio_numsize_match_title_exact', lambda row :row['numsize_match_title_exact'] / (row['len_of_numsize_query']+1)),
    ('ratio_numsize_match_description_exact', lambda row :row['numsize_match_description_exact'] / (row['len_of_numsize_query']+1)),
])



if __name__=='__main__':
    import doctest
    doctest.testmod()
