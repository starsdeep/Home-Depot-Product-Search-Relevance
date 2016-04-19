from hyperopt import hp
import math
rfr1 = {
    'max_depth': hp.choice('max_depth', range(3,30)),
    'max_features': hp.choice('max_features', range(3,50)),
    'n_estimators': hp.choice('n_estimators', [300, 500,1000]),
}

ridge1= {
    #'alpha': hp.loguniform('alpha', math.log(0.0001), math.log(10)),
    'alpha': 0.2,
    'max_iter': 50000,
    'tol': 0.000001
    }

param_space_dict = {
    'rfr1': rfr1,
    'ridge1': ridge1,
    }


