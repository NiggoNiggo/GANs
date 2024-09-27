import os
import ast
import pandas as pd
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt





class RandomSearch():
    def __init__(self,
                 name:str,
                 epochs:int,
                 param_grid:dict,
                 params,
                 num_testings,
                 gan):
        self.epochs = epochs
        self.params = params
        self.params.epochs = epochs
        self.param_grid = param_grid
        self.upack_params = [x for x in self.param_grid.keys()]
        self.previous_params = {}
        self._num_testings = num_testings
        self.gan = gan
        self.name = name
        
        self.results = {}
    
    # @property
    # def num_testings(self):
    #     return self.num_testings
    
    # @num_testings.setter
    # def num_testings(self, num_testing):
    #     number = len([value for values in self.param_grid.values() for value in values])
    #     self._num_testings = min(num_testing, number)


    
    def tune_params(self, device: str = "cuda"):
        # save params
        previous_params = []
        results = []

        for num in range(self._num_testings):
            print(f"Testing iteration: {num}")

            #current params in a dict to save it
            current_params = {}
            for key, values in self.param_grid.items():
                value = random.choice(values)
                current_params[key] = value

            #change params in the params of the gan
            for key, value in current_params.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)

            # train gan for some epochs
            gan = self.gan(self.params, device, self.name)
            self.save_path = os.path.join(gan.params.save_path,gan.name,"optimization")
            gan.tune_params(self.epochs)

            # save results
            previous_params.append(current_params)
            results.append(gan.scores)

        # merge results and the params
        result_dict = {
            'params': previous_params,
            'results': results
        }
        self.plot_results(previous_params,results)
        df_results = self.get_results(result_dict)
        print(df_results)
    
        return result_dict
                

    

                


    def plot_results(self, 
                    params:list,
                    results:list):
        """plot_results Plot the results of the hyperparameter tuning with regresson lines

        Parameters
        ----------
        params : list
            list with the testest parameters in dict form 
        results : list
            contains the results in dict form 
        """
        # Definiere Farben und Marker
        colors = ["r", "g", "b", "c", "y", "m"]
        markers = ['o', 's', 'D', '^', 'v', '*']
        
        # Anzahl der Metriken bestimmen
        metric_keys = list(results[0].keys())
        num_metrics = len(metric_keys)
        
        # Erstelle Subplots für jede Metrik (untereinander)
        fig, ax = plt.subplots(nrows=num_metrics, ncols=1, figsize=(10, 5 * num_metrics))
        
        if num_metrics == 1:
            ax = [ax]

        # Für die legend Handles und Labels
        legend_handles = []
        legend_labels = []

        for idx_param, param in enumerate(params):
            for idx_metric, key in enumerate(metric_keys):
                metric_values = results[idx_param][key]
                x_loss = np.linspace(0, self.epochs, len(metric_values), endpoint=True)

                ax[idx_metric].set_title(key, fontsize=14)
                scatter = ax[idx_metric].scatter(x_loss, metric_values, 
                                                label=f"{param}", 
                                                color=colors[idx_param % len(colors)],
                                                marker=markers[idx_param % len(markers)], 
                                                s=100)
                
                trend_loss = self.get_trend(metric_values)
                ax[idx_metric].plot(x_loss, trend_loss, 
                                    label=f"Trend: {param}", 
                                    color=colors[idx_param % len(colors)],
                                    linestyle='--')
                
                # Füge Handles und Labels zur Legende hinzu
                legend_handles.append(scatter)
                legend_labels.append(f"{param}")

                ax[idx_metric].grid()
                ax[idx_metric].set_xlabel("Epochs")
                ax[idx_metric].set_ylabel("loss" if key != "fid" else "FID Score")

        # Speichern der Legende als PNG
        legend = plt.figure(figsize=(3, 5))  # Legend-Figur anpassen
        legend.legend(legend_handles, legend_labels, loc='center', fontsize=12)
        legend.savefig(os.path.join(self.save_path, 'legend.png'), bbox_inches='tight')
        plt.close(legend)
        
        # Platz für die Legende rechts schaffen
        plt.tight_layout(rect=[0, 0, 0.75, 1])  # Platz für die Legende
        plt.savefig(os.path.join(self.save_path, "Hyperparameter_tuning_result_improved.png"))
        plt.show()


    
    def get_results(self,all_scores:dict):
        """get_results receive the results from the hyperparameter tuning and analyse them.
        A plot is made and also a csv file is created to analyze the hyperparameter tuning.

        Parameters
        ----------
        all_scores : dict
            contains all scores and the additional parameters that were testet

        Returns
        -------
        dict 
            contains the information about the training as a dict
        """
    # DataFrame initialisieren
        results_dict = pd.DataFrame(columns=("Params","Loss", "Median", "Max", "Min", "Slope", "Error"))
        
        # itterate through the results
        for idx, result in enumerate(all_scores['results']):
            for name, scores in result.items():
                # name == fid, scores = array
                epochs = np.arange(len(scores))  # Epochs as x values
                params = all_scores["params"][idx]
                # linear regression with scipy 
                slope, intercept, r_value, p_value, std_err = stats.linregress(epochs, scores)
                
                # calcualtion of some stats
                median = np.median(scores)
                min_score, max_score = np.min(scores), np.max(scores)
                
                # row to add in the datafram
                row = {
                    "Params":params,
                    "Loss": name,
                    "Median": round(median,2),
                    "Max": round(max_score,2),
                    "Min": round(min_score,2),
                    "Slope": round(slope,3),
                    "Error": round(std_err,4)
                }
                
                # concat row to dataframe
                results_dict = pd.concat([results_dict, pd.DataFrame([row])], ignore_index=True)
            #save as a csv file for further processing
            results_dict.to_csv(os.path.join(self.save_path,"results_hypertuning.csv"),index=False)
            
    def check_conditions(self, 
                        filename:str):
        #read data 
        data = pd.read_csv(filename,index_col=False)
        #best params lowest error on loss g and loss d -negative slope at loss d and lowest fid score
        best_condition = False 
        #get fid rows
        min_fid = data[data['Loss'] == 'fid'].min()
        param_dict = ast.literal_eval(min_fid["Params"])
        for param,value in param_dict.items():
            if hasattr(self.params,param):
                setattr(self.params,param,value)
        return self.params
    

    def return_new_gan_settings(self):
        self.params


    @staticmethod
    def get_trend(loss_values:list)->np.ndarray:
        """get_trend calculate the linear regression of the given metric, important are 
        just the error and the slope to reconstruct the linear regression function.\n
        @static methode


        Parameters
        ----------
        loss_values : list
            contains the loss_values or the metric values

        Returns
        -------
        np.ndarray
            linear regression graph to the given metric
        """
        #iterate through every value in the results list
        epochs = np.arange(len(loss_values))     
        # calculate the linear regression  
        slope, intercept, r_value, p_value, std_err = stats.linregress(epochs, loss_values)
        # print(f"Slope: {slope:.2f},error: {std_err:.2f}")
        #return a regression linear function 
        return intercept + np.arange(len(loss_values))*slope

