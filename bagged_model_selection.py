from load_data import load_data
import sys, os
import json
import operator
import copy
from trials_helper import TrialsList, load_trials
from base_model import fmean_squared_error_
import random
from model_selection import average_selection

Trials = TrialsList()

df_all, num_train, num_test = load_data(-1)
df_train = df_all[:num_train]
df_test = df_all[:num_test]
y_train = df_train['relevance'].values
model_idx_dict = dict()

def load_data(config):
    global model_idx_dict
    global Trials
    for idx, dir_path in enumerate(config['model_library_path_list']):
        trial_result_list, train_pred_list, test_pred_list = load_trials(dir_path)
        print("load dir %s done, model library size %d" % (dir_path, len(trial_result_list)))
        Trials.append(trial_result_list, train_pred_list, test_pred_list)
        model_name = config['model_names'][idx] 
        model_idx_dict[model_name] = (len(Trials.train_pred_list) - len(trial_result_list), len(Trials.train_pred_list))


def sample_with_ratio(ratio, idx_start, idx_end):
    k = int((idx_end - idx_start) * ratio)
    return random.sample(range(idx_start, idx_end), k)


def get_random_model_index(model_ratio_dict, model_idx_dict):
    random_index = []
    for key in model_ratio_dict.keys():
        tmp_ratio = model_ratio_dict[key]
        (tmp_idx_start, tmp_idx_end) = model_idx_dict[key]
        tmp_random_index = sample_with_ratio(tmp_ratio, tmp_idx_start, tmp_idx_end)
        random_index += tmp_random_index
        print("sample from %s, index range: %d, %d, ratio: %f, sampled %d" % (key, tmp_idx_start, tmp_idx_end, tmp_ratio, len(tmp_random_index)))
    return sorted(random_index)


def bag_model_selection(bag_num , config):
    global model_idx_dict
    global Trials
    model_ratio_dict = config['model_ratio_dict']
    for i in range(20):
        print("\nbag iteration %d" % i)
        random_model_index = get_random_model_index(model_ratio_dict, model_idx_dict)
        trial_result_list, avg_rmse_list, index_list =  average_selection(y_train, Trials, config['max_ensemble'], config['replace'], random_model_index)
        with open(os.path.join(output_path, 'ensemble_avg_model_list_%d.json' % i), 'w') as outfile:
            json.dump(trial_result_list, outfile)
        with open(os.path.join(output_path, 'ensemble_avg_rmse_list_%d.json' % i), 'w') as outfile:
            json.dump(avg_rmse_list, outfile)
        with open(os.path.join(output_path, 'ensemble_avg_index_list_%d.json' % i), 'w') as outfile:
            json.dump(index_list, outfile)



if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("<output directory>")
        sys.exit()

    output_path = sys.argv[1]
    with open(os.path.join(sys.argv[1], 'config.json')) as infile:
        config = json.load(infile)
    load_data(config)
    bag_model_selection(20, config)


