import pandas as pd

def get_average(dic):
    rsum, num = 0, 0
    for key, value in dic.items():
        rsum += value[0]
        num += value[1]
    return rsum / num

def apply_hist_rel(df, count_dict, name):
    new_feature_name = name + '_hist_rel'
    avg = get_average(count_dict)    
    for index, row in df.iterrows():
        key = row[name]
        if key in count_dict:
            relsum, count = count_dict[key]
        else:
            df.loc[index, new_feature_name] = avg
            continue
        if 'relevance' not in row:
            df.loc[index, new_feature_name] = relsum/count
        elif count==1:
            df.loc[index, new_feature_name] = avg
        else:
            df.loc[index, new_feature_name] = (relsum-row['relevance'])/(count-1)

def add_hist_rel(df_train, df_test, name):
    count_dict = dict()
    for index, row in df_train.iterrows():
        key = row[name]
        if key not in count_dict:
            count_dict[key] = [row['relevance'], 1]
        else:
            count_dict[key][0] += row['relevance']
            count_dict[key][1] += 1
    print(name + ': ' + str(len(count_dict)))
    apply_hist_rel(df_train, count_dict, name)
    apply_hist_rel(df_test, count_dict, name)

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")

add_hist_rel(df_train, df_test, 'product_uid')
add_hist_rel(df_train, df_test, 'search_term')

df_train.to_csv("../input/train_histrel.csv", encoding="utf8")
df_test.to_csv("../input/test_histrel.csv", encoding="utf8")
