import sys, os
import pandas as pd

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("<features_path><output_path>")
    df = pd.read_csv(sys.argv[1], encoding="ISO-8859-1", index_col=0)
    for feature in df.columns.values:
        tmp_df = df[[feature]]
        tmp_df.to_csv(os.path.join(sys.argv[2], feature + '.csv'), encoding="utf8")