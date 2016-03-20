import pandas as pd
import time


def load_data(num_sample=-1):
    start_time = time.time()

    df_train = pd.read_csv('./input/train.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('./input/test.csv', encoding="ISO-8859-1")
    if num_sample > 0:
        df_train = df_train[:num_sample]
        df_test = df_test[:num_sample]
    df_pro_desc = pd.read_csv('./input/product_descriptions.csv')
    df_attr = pd.read_csv('./input/attributes.csv')
    df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')

    num_train = df_train.shape[0]
    num_test = df_test.shape[0]
    print("load %d training sample, load %d test sample" % (num_train, num_test))
    print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time) / 60), 2))

    return df_all, num_train, num_test


