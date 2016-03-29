import model

class ModelFactory(object):

    def __init__(self):
        self.model_dict = {
            'rfr': model.RandomForestRegression(),
            'rfc': model.RandomForestClassification(),
            'xgbr': model.XgboostRegression(),
            'gbdtr': model.GbdtRegression(),
            'multi': model.MultiClassifier()
        }

    def create_model(self, config):
        model = self.model_dict[config['model']]
        model.set_config(config)
        return model
