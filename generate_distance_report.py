from scipy.spatial import distance
from base_model import fmean_squared_error_
import os, sys
from load_data import load_data
from trials_helper import *
from base_model import fmean_squared_error_

model_idx_dict = dict()

def report_distance(trials, y_train, best_test_pred, best_train_pred, model_name):
    (idx_start, idx_end) = model_idx_dict[model_name]
    print(model_name + ", idx range: %d, %d"  % (idx_start, idx_end) )
    test_distance_list = []
    train_distance_list = []
    for i in range(idx_start, idx_end):
        test_d = distance.correlation(best_test_pred, trials.test_pred_list[i])
        train_d = distance.correlation(best_train_pred, trials.train_pred_list[i])
        test_distance_list.append(test_d)
        train_distance_list.append(train_d)
    sorted_list = sorted(range(len(test_distance_list)), key=lambda k: test_distance_list[k], reverse=True)
    print('%12s  %12s  %12s  %12s  %12s' % ("corr_test", "corr_train", "self rmse", "avg rmse", "index"))
    for idx in sorted_list:
        avg_train_pred = (best_train_pred + trials.train_pred_list[idx_start + idx]) / 2.0
        avg_rmse = fmean_squared_error_(y_train, avg_train_pred)
        print('%12f  %12f  %12f  %12f  %12d' % (test_distance_list[idx], train_distance_list[idx], trials.trial_result_list[idx_start + idx]['loss'], avg_rmse, (idx_start + idx) )) 

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("<output directory>")
        sys.exit()
    
    output_path = sys.argv[1]
    with open(os.path.join(sys.argv[1], 'config.json')) as infile:
        config = json.load(infile)
    
    # load data
    df_all, num_train, num_test = load_data(-1)
    df_train = df_all[:num_train]
    df_test = df_all[:num_test]
    y_train = df_train['relevance'].values
    Trials = TrialsList()
    i = 0
    for dir_path in config['model_library_path_list']:
        tmp_trial_result_list, tmp_train_pred_list, tmp_test_pred_list = load_trials(dir_path)
        print("load dir %s done, model library size %d" % (dir_path, len(tmp_trial_result_list)))
        Trials.append(tmp_trial_result_list, tmp_train_pred_list, tmp_test_pred_list)
        model_name = dir_path.split('/')[-2]
        model_idx_dict[model_name] = (len(Trials.train_pred_list) - len(tmp_trial_result_list), len(Trials.train_pred_list))
        

    index, trial_result, train_pred, test_pred = Trials.best_trial(verbose=True)

    for key in model_idx_dict.keys():
        report_distance(Trials, y_train, test_pred, train_pred, key)        
    
    

    """
    print("rmse %f\n,model is %s\n" % (rmse_list[-1], trial_result_list[-1])) 
    while(1):
        model_to_view = input("enter which model you want to inspect with current ensemble result.(%s): " % " ".join(model_idx_dict.keys()))
        report_distance(Trials, avg_train_pred, model_to_view)
        index_to_choose = int(input("choose the the index of model, that you want to blend into current ensemble: "))
        model_to_blend = Trials.train_pred_list[index_to_choose]
        
        avg_train_pred = (avg_train_pred + model_to_blend) / 2.0
        avg_train_pred = (avg_train_pred + model_to_blend) / 2.0
        
        rmse = fmean_squared_error_(y_train, avg_train_pred)
        rmse_list.append(rmse)
        trial_result_list.append(Trials.trial_result_list[index_to_choose])
        with open(os.path.join(sys.argv[1], 'interactive_model_list.json'), 'w') as outfile:
            json.dump(outfile, trial_result_list)
        print("blend done. new ensemble rmse: %f" % rmse)
   """ 
