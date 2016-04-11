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

    with open(os.path.join(sys.argv[1], 'config.json')) as infile:
        config = json.load(infile)

    # feature extraction
    df_train, df_test = get_feature(config)
    id_test = df_test['id']
    y_train = df_train['relevance'].values
    X_train = df_train[:]
    X_test = df_test[:]

    #model
    start_time = time.time()
    mf = ModelFactory()
    model = mf.create_model(config)
    model.fit(X_train, y_train)
    df_all = pd.concat((df_train, df_test), axis=0)
    y_pred_all = model.predict(df_all)
    y_pred = model.predict(X_test)
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(os.path.join(sys.argv[1],'submission.csv'),index=False)
    pd.DataFrame({"id": df_all['id'], "relevance": y_pred_all}).to_csv(os.path.join(sys.argv[1],'predict_relevance.csv'),index=False)
