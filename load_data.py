import pandas as pd
import time
from functools import partial
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def load_data(num_sample=-1):
    start_time = time.time()
    df_train = pd.read_csv('./input/train.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('./input/modify_test.csv', encoding="ISO-8859-1")
    if num_sample > 0:
        df_train = df_train[:num_sample]
        df_test = df_test[-num_sample:]
    df_pro_desc = pd.read_csv('./input/product_descriptions.csv')
    df_attr = pd.read_csv('./input/attributes.csv')
    df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
    
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    
    # fill empty brand
    print('--- Fill Empty Brand based on Title Similarity ---')
    df_all['brand'] = df_all['brand'].fillna('Unbranded')
    brands_set = set(df_all['brand'].values)
    brands_set.remove('Unbranded')
    df_all['brand'] = df_all.apply( partial(fill_unbrand, brands_set), axis=1)
    
    num_train = df_train.shape[0]
    num_test = df_test.shape[0]
    print("load %d sample, %d training sample, %d test sample" % (df_all.shape[0], num_train, num_test))
    print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
    return df_all, num_train, num_test


def fill_unbrand(brands_set, row ):
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b')
    analyzer = bigram_vectorizer.build_analyzer()
    if row['brand'] == 'Unbranded':
        ngrams = analyzer(row['product_title'])
        ngrams.reverse()
        for item in ngrams:
            if item in brands_set:
                return item
        return np.nan
    return row['brand']
            

if __name__ == '__main__':
    load_data(1000)