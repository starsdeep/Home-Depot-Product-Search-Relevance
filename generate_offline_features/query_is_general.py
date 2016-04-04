

# steps to use this module:
# no need to run this module, I add the feature to input/test.csv and input/train.csv

__author__ = 'fucus'
import config.project as project
import pandas as pd
import utility
import math
import feature


query_is_general_file = open("%s/%s/query_is_general.csv" % (project.project_path, feature.feature_path), "w", encoding="utf8")

df_train = pd.read_csv(project.original_train_file, encoding="ISO-8859-1")
df_test = pd.read_csv(project.original_test_file, encoding="ISO-8859-1")
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
query_is_general = {}
query_to_relevance_count_list = {}
min_general_count = 4
query_is_general_feature = []
general_query_count = 0

for index, item in df_train.iterrows():
    relevance = int(item["relevance"])
    search_term = utility.str_stem(item["search_term"])
    if search_term not in query_to_relevance_count_list.keys():
        query_to_relevance_count_list[search_term] = [0, 0, 0]
    query_to_relevance_count_list[search_term][relevance-1] += 1

for key in query_to_relevance_count_list.keys():
    if query_to_relevance_count_list[key][0] == 0 and query_to_relevance_count_list[key][2] == 0 and query_to_relevance_count_list[key][1] > min_general_count:
        query_is_general[key] = True

for index, item in df_all.iterrows():
    search_term = utility.str_stem(item["search_term"])
    if search_term in query_is_general.keys():
        query_is_general_feature.append(query_is_general[search_term])
        if query_is_general[search_term]:
            general_query_count += 1
    else:
        query_is_general_feature.append(False)


print("%d/%d querys are general" % (general_query_count, len(df_all)))
print("%d/%d in train.csv are general" % (len(query_is_general), len(query_to_relevance_count_list)) )

pd.DataFrame(query_is_general_feature, columns=["query_is_general", ])\
    .to_csv("%s/%s/query_is_general.csv" % (project.project_path, feature.feature_path))

for key in query_to_relevance_count_list.keys():
    if key in query_is_general.keys() and query_is_general[key]:
        print(key)