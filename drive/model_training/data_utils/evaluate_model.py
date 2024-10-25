import numpy as np 
import pandas as pd

from metric_energy import * 
from drive.model_training.models.kinematic.ideal_diff_drive import Ideal_diff_drive


PATH_TO_MODEL_EVALUATION  = pathlib.Path('drive/model_training/data_utils/metric_config.yaml')


class ModelEvaluationError():


    def __init__(self,metric_name) -> None:
            

            with open(PATH_TO_METRIC, 'r') as file:
                metric_param_config = yaml.safe_load(file)
            

            for metric,params_metric in metric_param_config["metric"].items():

                if metric == metric_name:

                    self.metric_parameters = params_metric
            

            self.metric_name = metric_name
            

            # Read results_file 
            list_possible_metric = ["kinetic_energy",]
            if not PATH_TO_RESULT_FILE.is_file():
                empty_dict  = {}
                for metric in list_possible_metric:
                    empty_dict[metric] = {}
        
                with open(PATH_TO_RESULT_FILE, 'wb') as file:
                    results_file = pickle.dump(empty_dict,file)
            

            with open(PATH_TO_RESULT_FILE, 'rb') as file:
                self.results_file = pickle.load(file)
            
            