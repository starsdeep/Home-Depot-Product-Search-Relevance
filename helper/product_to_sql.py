__author__ = 'fucus'
import pandas as pd
import config.project as project

df_train = pd.read_csv(project.original_train_file, encoding="ISO-8859-1")
df_pro_desc = pd.read_csv(project.original_product_descriptions)
df_attr = pd.read_csv(project.original_product_descriptions)
