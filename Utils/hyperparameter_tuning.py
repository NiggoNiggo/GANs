import os
import pandas as pd
import random
import numpy as np


from Base_Models.gan_base import GanBase


class RandomSearch():
    def __init__(self,
                 epochs:int,
                 param_grid:dict,
                 params,
                 num_testings,
                 gan):
        self.epochs = epochs
        self.params = params
        self.params.epochs = epochs
        self.param_grid = param_grid
        self.previous_params = {}
        self._num_testings = num_testings
        self.gan = gan
        self.results = {}
    
    # @property
    # def num_testings(self):
    #     return self.num_testings
    
    # @num_testings.setter
    # def num_testings(self, num_testing):
    #     number = len([value for values in self.param_grid.values() for value in values])
    #     self._num_testings = min(num_testing, number)


    
    def tune_params(self, name: str, device: str = "cuda"):
        # Erstelle ein Dictionary zur Speicherung der Parameter und Ergebnisse
        previous_params = []
        results = []

        for num in range(self._num_testings):
            print(f"Testing iteration: {num}")

            # Definiere das aktuelle Param-Dictionary
            current_params = {}
            for key, values in self.param_grid.items():
                value = random.choice(values)
                current_params[key] = value

            # Aktualisiere die Parameter
            for key, value in current_params.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)

            # Erstelle und trainiere das GAN
            gan = self.gan(self.params, device, name)
            gan.tune_params(self.epochs)

            # Speichere die Ergebnisse
            previous_params.append(current_params)
            results.append(gan.scores)

        # Kombiniere die Ergebnisse und Parameter
        result_dict = {
            'params': previous_params,
            'results': results
        }

        # Drucke die Ergebnisse
        for idx, (param_set, score_set) in enumerate(zip(previous_params, results)):
            print(f"Test {idx}: Parameters: {param_set}, Mean Score loss_d: {np.mean(score_set['loss_d'])}")

        return result_dict
                
    
    def find_best_result(self):
        #iterate through results and search for index and find it by the settings dict
        for key, values in self.results.items():
            if key == "fid":
                min_index = np.argmin(values)
                # np.




    def save_stats(self):
        df = pd.DataFrame(self.results)
        path = os.path.join(self.gan.save_path,"optimization","results.csv")
        df.to_csv(path,index=False)

    def __call__(self,x):
        return x







class BaysesianOptimization:
    def __init__(self,
                 gan):
        self.gan = gan

    def get_objective(self):
        # self.gan.
        pass














class GridSearch:
    def __init__(self):
        pass
    def __call__(self,x):
        return x

class BaysianSearch:
    def __init__(self):
        pass
    def __call__(self,x):
        return x

                
if __name__ == "__main__":
    pass
    data = {
            "lr":[1e-5,1e-4],
            "bs":[64,128],
            "n_crit":[5,10]
            }    