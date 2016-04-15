__author__ = 'starsdeep'
import os
import json

train_pred_filename_tpl = 'train_pred_trial_%d.csv'
trails_filename = 'hyperopt_trials.json'

class TrialsHelper():

    def __init__(self, dir_path):
        self.dir_path = dir_path
        with open(os.path.join(dir_path, trails_filename)) as infile:
            self.trials = json.load(infile)
        self.sorted_trial_counters = sorted(range(len(self.trials)), key=lambda k: self.trials[k]['loss'])


    def get_nth_train_pred(self, n):
        file = os.path.join(self.dir_path, train_pred_filename_tpl % n)
        df_train_pred = pd.read_csv(file, encoding="ISO-8859-1", index_col=0)

    def get_best_trail_counter(self):
        return self.sorted_trial_counters[0]

    def get_best_train_pred(self):
        return self.get_nth_train_pred(self.sorted_trial_counters[0])



