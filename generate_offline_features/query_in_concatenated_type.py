__author__ = 'fucus'

# generate feature file: word_in_concatenated_type.csv

import config.project as project
import pandas as pd
import feature
import utility
df_train = pd.read_csv(project.original_train_file, encoding="ISO-8859-1")
df_test = pd.read_csv(project.original_test_file, encoding="ISO-8859-1")
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

query_is_general_feature = []
word_in_concatenated_type = []


product_uid_to_concatenated_type = {}

df_attr = pd.read_csv(project.original_attr_file)
count = 0
for index, item in df_attr.iterrows():
    count += 1
    if count % 1000 == 0:
        print("calculated %d/2044804 lines" % count)
    try:
        product_uid = int(item['product_uid'])
    except:
        print("%s cannot converted to int" % str(item['product_uid']))
        continue
    raw_name = str(item['name'])
    if raw_name[-4:] == 'Type':
        value = feature.search_term_clean(str(item['value']))
        name = feature.search_term_clean(raw_name)
        if product_uid not in product_uid_to_concatenated_type:
            product_uid_to_concatenated_type[product_uid] = value
        else:
            product_uid_to_concatenated_type[product_uid] += " %s" % value

for index, item in df_all.iterrows():
    search_term = feature.search_term_clean(str(item['search_term']))
    try:
        product_uid = int(item['product_uid'])
    except:
        print("%s cannot converted to int" % str(item['product_uid']))
        continue

    if product_uid in product_uid_to_concatenated_type.keys():
        count = utility.num_common_word(search_term, product_uid_to_concatenated_type[product_uid])
    else:
        count = 0
    word_in_concatenated_type.append(count)



print("%d search terms are in concatenated_type" % sum([1 for x in word_in_concatenated_type if x > 0]))
pd.DataFrame(word_in_concatenated_type, columns=["word_in_concatenated_type", ])\
    .to_csv("%s/%s/word_in_concatenated_type.csv" % (project.project_path, feature.feature_path))
