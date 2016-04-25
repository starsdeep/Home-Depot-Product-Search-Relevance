
import os,sys
from trials_helper import load_trial_result_list

if __name__ == '__main__':

    if len(sys.argv)!=2:
        print("<dir_path>")
        sys.exit()
    dir_path = sys.argv[1]

    trial_result_list = load_trial_result_list(dir_path)
    sorted_list = sorted(range(len(trial_result_list)), key=lambda k: trial_result_list[k]['loss'])
    with open(os.path.join(dir_path, 'sorted_model_result.txt'), 'w') as outfile:
        for idx in sorted_list:
            outfile.write('index: %d, rmse: %f\n' % (idx, trial_result_list[idx]['loss']))
        outfile.close()
    print("write report to %s in %s/" % ('sorted_model_result.txt', dir_path))
