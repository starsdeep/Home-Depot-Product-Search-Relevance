

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

query_is_general = {}

for index, item in df_train.iterrows():
    relevance = int(item["relevance"])
    search_term = utility.str_stem(item["search_term"])

    if relevance != 2:
        query_is_general[search_term] = False
    elif search_term not in query_is_general.keys():
        query_is_general[search_term] = True

query_is_general_feature = []

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)


general_query_count = 0
for index, item in df_train.iterrows():
    search_term = utility.str_stem(item["search_term"])
    if search_term in query_is_general.keys():
        query_is_general_feature.append(query_is_general[search_term])
        if query_is_general[search_term]:
            general_query_count += 1
    else:
        query_is_general_feature.append(False)


print("%d querys are general" % general_query_count)
pd.DataFrame(query_is_general_feature, columns=["query_is_general", ])\
    .to_csv("%s/%s/query_is_general.csv" % (project.project_path, feature.feature_path))