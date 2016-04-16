from load_data import load_data
import sys, os
import json
import operator
import copy
from trials_helper import TrialsList, load_trials
from base_model import fmean_squared_error_

output_path = ""

def average_selection(y_train, Trials, max_ensemble=20):
    print("model selection using average cross validation score")
    print("model library size %d" % len(Trials.trial_result_list))
    n_ensemble = min(max_ensemble, len(Trials.trial_result_list))


    index, trial_result, train_pred = Trials.get_best_trial(verbose=True)
    avg_rmse_list = []
    avg_rmse_list.append(trial_result['loss'])
    trial_result_list = []
    trial_result_list.append(trial_result)
    avg_train_pred = train_pred
    i = 1
    print("\n======= ensemble %d, rmse %f=========\n%s" % (i, avg_rmse_list[i-1], trial_result_list[i-1]))
    while i < n_ensemble:
        avg_train_pred *= (i/(i+1))
        tmp_avg_rmse_list = []
        for train_pred in Trials.train_pred_list:
            tmp_avg = avg_train_pred + train_pred / (i+1.0)
            # print(str(tmp_avg[:10]))
            tmp_avg_rmse_list.append(fmean_squared_error_(y_train, tmp_avg))
        # print(str(tmp_avg_rmse_list))
        min_index, min_value = min(enumerate(tmp_avg_rmse_list), key=operator.itemgetter(1))
        avg_train_pred = avg_train_pred + Trials.train_pred_list[min_index] / (i+1.0)
        avg_rmse_list.append(min_value)
        best_trials = copy.deepcopy(Trials.trial_result_list[min_index])
        Trials.del_n_trial_(min_index)
        trial_result_list.append(Trials.trial_result_list[min_index])
        i += 1
        print("\n======= ensemble %d, rmse %f=========\n%s\n" % (i, avg_rmse_list[i-1], trial_result_list[i-1]))

    #save result
    with open(os.path.join(output_path, 'ensemble_avg_model_list.json'), 'w') as outfile:
        json.dump(trial_result_list, outfile)
    with open(os.path.join(output_path, 'ensemble_avg_rmse_list.json'), 'w') as outfile:
        json.dump(avg_rmse_list, outfile)


def linear_selection():
    pass


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("<output directory>")
        sys.exit()

    output_path = sys.argv[1]
    with open(os.path.join(sys.argv[1], 'config.json')) as infile:
        config = json.load(infile)

    df_all, num_train, num_test = load_data(-1)
    df_train = df_all[:num_train]
    df_test = df_all[:num_test]
    y_train = df_train['relevance'].values

    Trials = TrialsList()
    for dir_path in config['model_library_path_list']:
        print("loading dir " + dir_path)
        trial_result_list, train_pred_list = load_trials(dir_path)
        Trials.append(trial_result_list, train_pred_list)


    average_selection(y_train, Trials, max_ensemble=6)



