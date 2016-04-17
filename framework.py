import hashlib
import pandas as pd
import random
random.seed(2016)
from load_data import load_data
from feature import get_feature
from feature import load_feature
import sys
import os
import json
import time
from model_factory import ModelFactory

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("<output directory>")
        sys.exit()

    start_time = time.time()
    print("---------------------start---------------------------")
    print("start at %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(sys.argv[1], 'config.json')) as infile:
        config = json.load(infile)

    # feature extraction
    df_all, num_train, num_test = get_feature(config)
    df_train = df_all[:num_train]
    df_test = df_all[-num_test:]
    id_test = df_test['id']

    y_train = df_train['relevance'].values

    #model
    mf = ModelFactory()
    model = mf.create_model(config)

    X_all, column_names = model.feature_union(df_all)
    X_train = X_all[:num_train]
    X_test = X_all[-num_test:]

    model.fit(X_train, y_train, df_train, column_names)
    if config['model']=='multi':
        y_pred = model.predict(df_test)
    else:
        y_pred = model.predict(X_test)
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(os.path.join(sys.argv[1],'submission.csv'),index=False)
    print("---------------------end---------------------------")
    print("\n\n\n\n")