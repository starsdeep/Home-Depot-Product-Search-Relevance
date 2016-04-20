import model

class ModelFactory(object):

    def __init__(self):
        self.model_dict = {
            'etr': model.ExtraTreesRegression(),
            'rfr': model.RandomForestRegression(),
            'rfc': model.RandomForestClassification(),
            'xgbr': model.XgboostRegression(),
            'gbdtr': model.GbdtRegression(),
            'multi': model.MultiClassifier(),
            'three': model.ThreePartRandomForestClassification(),
            'ridger': model.RidgeRegression(),
            'linear': model.LinearRegression(),
            'lassor': model.LassoRegression(),
            'rgf': model.RegularizedGreedyForest(),
            'svr': model.SVR()
        }

    def create_model(self, config):
        model = self.model_dict[config['model']]
        model.set_config(config)
        return model
