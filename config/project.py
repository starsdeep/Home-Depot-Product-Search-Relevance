__author__ = 'fucus'

import os
import re

project_path = re.search(r'.*Home_Depot_Product_Search_Relevance[^\/]*',os.getcwd()).group()
original_train_file = "%s/input/train.csv" % project_path
original_test_file = "%s/input/test.csv" % project_path
original_product_descriptions = "%s/input/product_descriptions.csv" % project_path

original_attr_file = "%s/input/attributes.csv" % project_path

original_train_file_part = "%s/input/train_top_100.csv" % project_path
original_test_file_part = "%s/input/test_top_100.csv" % project_path

train_file_split_part_prefer_low_score = "%s/input/train_prefer_low.csv" % project_path
train_file_split_part_prefer_mid_score = "%s/input/train_prefer_mid.csv" % project_path
train_file_split_part_prefer_high_score = "%s/input/train_prefer_high.csv" % project_path

submission_merged_result = "%s/output/rfc_chenqiang_merge_three_part_2/submission.csv" % project_path

add_bm25_train_path = "%s/generate_offline_features/add_bm25_train.csv" % project_path
add_bm25_test_path = "%s/generate_offline_features/add_bm25_test.csv" % project_path
