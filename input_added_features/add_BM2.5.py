
# implemented algorithm description in https://en.wikipedia.org/wiki/Okapi_BM25

# steps to use this module:
# no need to run this module, I add the feature to input/test.csv and input/train.csv

__author__ = 'fucus'
import config.project as project
import pandas as pd
import utility
import math


def cal_idf(N, count):
    """

    :param N: documents count
    :param count: how many documents include word
    :return:
    """
    return math.log(N - count + 0.5) / (count + 0.5)


def cal_bm25(idf, word_frequency, len_of_doc, average_len, k1, b):
    """
    :param idf:
    :param word_frequency:
    :param len_of_doc: length of documents
    :param average_len: average length of documents
    :param k1:
    :param b:
    :return:
    """
    return idf * (word_frequency * (k1 + 1)) / (word_frequency + k1 * (1 - b + b * len_of_doc * 1.0 / average_len))


df_train = pd.read_csv(project.original_train_file, encoding="ISO-8859-1")
df_test = pd.read_csv(project.original_test_file, encoding="ISO-8859-1")

BM25_title_train = []
BM25_description_train = []

BM25_title_test = []
BM25_description_test = []

# an advanced optimization k1 from [1.2, 2.0], b = .75
BM25_arg_k1 = 1.2
BM25_arg_b = 0.75

df_train_descriptions = pd.read_csv(project.original_product_descriptions)


word_title_count = {}
word_description_count = {}

word_in_title_idf = {}
word_in_description_idf = {}
title_idf_unknown_word = 0
description_idf_unknown_word = 0

# index with product_uid, next index is the word
word_frequency_in_title = {}
word_frequency_in_description = {}
title_length = {}
description_length = {}

title_avg_length = 0
description_avg_length = 0

title_count = 0
description_count = 0

# calculate the IDF

product_title = {}
for index, item in df_train.iterrows() :
    if item["product_uid"] not in product_title.keys():
        product_title[item["product_uid"]] = item["product_title"]

for index, item in df_test.iterrows():
    if item["product_uid"] not in product_title.keys():
        product_title[item["product_uid"]] = item["product_title"]

for key in product_title.keys():
    title = product_title[key]
    word_frequency_in_title[key] = {}
    stem_title = utility.str_stem(title, by_pos_tag=False)
    title_length[key] = len(stem_title)
    for word in stem_title.split():
        if word in word_frequency_in_title[key]:
            word_frequency_in_title[key][word] += 1
        else:
            word_frequency_in_title[key][word] = 1
            if word in word_title_count.keys():
                word_title_count[word] += 1
            else:
                word_title_count[word] = 1

for index, item in df_train_descriptions.iterrows():
    key = item["product_uid"]
    stem_description = utility.str_stem(item["product_description"], by_pos_tag=False)
    word_frequency_in_description[key] = {}
    description_length[key] = len(stem_description)
    for word in stem_description.split():
        if word in word_frequency_in_description[key]:
            word_frequency_in_description[key][word] += 1
        else:
            word_frequency_in_description[key][word] = 1
            if word in word_description_count.keys():
                word_description_count[word] += 1
            else:
                word_description_count[word] = 1

title_count = len(product_title) + 1
description_count = len(df_train_descriptions) + 1

title_avg_length = sum(title_length) / title_count + 1
description_avg_length = sum(description_length) / description_count + 1

for word in word_title_count.keys():
    word_in_title_idf[word] = cal_idf(title_count, word_title_count[word])

for word in word_description_count.keys():
    word_in_description_idf[word] = cal_idf(description_count, word_description_count[word])


description_idf_unknown_word = cal_idf(description_count, 0)
title_idf_unknown_word = cal_idf(title_count, 0)


def cal_bm25_query_with_title_and_description(df):
    BM25_title = []
    BM25_description = []
    for index, item in df.iterrows():
        pid = item["product_uid"]
        term = utility.str_stem(item["search_term"], by_pos_tag=False)
        title_score = 0
        description_score = 0
        for word in term.split():
            if word in word_in_title_idf.keys() and word in word_frequency_in_title[pid].keys():
                title_score += cal_bm25(word_in_title_idf[word], word_frequency_in_title[pid][word], title_length[pid]
                                        , title_avg_length, BM25_arg_k1, BM25_arg_b)
            if word in word_in_description_idf.keys() and word in word_frequency_in_description[pid].keys():
                description_score += cal_bm25(word_in_description_idf[word], word_frequency_in_description[pid][word]
                                             , description_length[pid], description_avg_length, BM25_arg_k1, BM25_arg_b)

        BM25_title.append(title_score)
        BM25_description.append(description_score)
    return BM25_title, BM25_description

BM25_title_train, BM25_description_train = cal_bm25_query_with_title_and_description(df_train)
BM25_title_test, BM25_description_test = cal_bm25_query_with_title_and_description(df_test)


df_train["title_query_BM25"] = BM25_title_train
df_train["description_query_BM25"] = BM25_description_train

df_test["title_query_BM25"] = BM25_title_test
df_test["description_query_BM25"] = BM25_description_test

df_train.to_csv(project.add_bm25_train_path)
df_test.to_csv(project.add_bm25_test_path)