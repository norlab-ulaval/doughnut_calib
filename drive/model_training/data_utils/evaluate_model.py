import numpy as np 
import pandas as pd

from metric_energy import * 
from drive.model_training.models.kinematic.ideal_diff_drive import Ideal_diff_drive


from drive.model_training.data_utils.metric_energy import *

PATH_TO_MODEL_EVALUATION_PARAM  = pathlib.Path('drive/model_training/data_utils/metric_config.yaml')


class ModelPredictionErrorComputer():

    def __init__(self,motion_model_name,robot_name,dataset) -> None:
            
        # Extract from dataset_relevant information:
        self.dataset2eval = dataset
        self.dt = 1/self.dataset2eval.rate
        
        with open(PATH_TO_MODEL_EVALUATION_PARAM, 'r') as file:
            metric_param_config = yaml.safe_load(file)["motion_model"]
        
        #############################################
        ########### LOAD THE MOTION MODEL ###########
        #############################################
        self.idd_name = "ideal_diff_drive"
        self.motion_model_name = motion_model_name
        if motion_model_name in list(metric_param_config.keys()):

            if motion_model_name == self.idd_name:
                
                motion_model_params = metric_param_config[self.idd_name]
                path_to_robot_param = motion_model_params["robot_rel_path"]
                with open(path_to_robot_param, 'r') as file:
                    robot_param = yaml.safe_load(file)

                if robot_name in list(robot_param["robot"].keys()):   
                    
                    idd_params  = robot_param["robot"][robot_name]
                    self.motion_model = Ideal_diff_drive(idd_params["wheel_radius"],idd_params["basewidth"],self.dt)
                    
                else:
                    raise ValueError("The robot {} is not available for the metric ideal_diff_drive ")
            else:
                raise ValueError("THere is a lack of adequation bwetween the if statement motion model and in the config")    
        else: 
            raise ValueError(f"The motion_model_name is not inf the config file \n {PATH_TO_MODEL_EVALUATION_PARAM}")
        self.metric_name = motion_model_name
        

        #############################################
        ########## Read/Create Results File #########
        #############################################
        self.path_to_res = pathlib.Path(metric_param_config["path_to_motion_model_results"])

        list_implemented_metric = ["MRMSE_2d"]

        if not self.path_to_res.is_file():
            
            empty_dict  = {}
            for metric in list_implemented_metric:
                empty_dict[metric] = {}
    
            with open(self.path_to_res, 'wb') as file:
                self.results_file = pickle.dump(empty_dict,file)
        
        # Load the results file. 
        with open(self.path_to_res, 'rb') as file:
            self.results_file = pickle.load(file)
        
    
    def evaluate_horizon_MRMSE_2d(self,dico_res,horizon_index):

        current_state = np.zeros((6,))
        u_matrix_wheel = np.array([dico_res["cmd_left_wheel"][horizon_index,:],dico_res["cmd_right_wheel"][horizon_index,:]])

        results_states = [current_state]

        h = u_matrix_wheel.shape[1]

        zeroes = np.zeros((h,))
        reference_state = np.array([dico_res["gt_body_lin_vel"][horizon_index,:],
                                    dico_res["gt_body_y_vel"][horizon_index,:],
                                    zeroes, # TODO: change for adding the z, roll and pitch dimension
                                    zeroes,
                                    zeroes,
                                    dico_res["gt_body_yaw_vel"][horizon_index,:]]).T
        for step in range(h-1):

            current_state = self.motion_model.predict(current_state, u_matrix_wheel[:,step])
            
            results_states.append(current_state)

        predictions = np.array(results_states)
        
        translation_MRMSE_2d = 1/h * np.sum(np.linalg.norm(predictions[:,0:2] - reference_state[:,0:2],axis=1))
        yaw_MRMSE_2d = 1/h * np.sum(np.sqrt((predictions[:,-1]-reference_state[:,-1])**2))
        #print(yaw_MRMSE_2d)

        return translation_MRMSE_2d, yaw_MRMSE_2d
    

    def evaluate_model_on_all_terrains_MRMSE_2d(self):

        list_terrain = list(self.dataset2eval.df.terrain.unique())


        list_df = []
        for terrain in list_terrain:
            
            terrain_translation_MRMSE_2d = []
            terrain_rotation_MRMSE_2d = []

            dico_info = self.dataset2eval[terrain]

            shape = dico_info["cmd_left_wheel"].shape

            for horizon_index in range(shape[0]):
                translation_MRMSE_2d, rotation_MRSME = self.evaluate_horizon_MRMSE_2d(dico_info,horizon_index)
                terrain_translation_MRMSE_2d.append(translation_MRMSE_2d)
                terrain_rotation_MRMSE_2d.append(rotation_MRSME)

            # Add filter that needs to be multiplied

            dico_update = {"translation_error":terrain_translation_MRMSE_2d,
                        "rotation_error":terrain_rotation_MRMSE_2d}
            
            
            for key,value in dico_info["col_2_multiply_and_add"].items():
                dico_update[key] = [value] * shape[0]
            
            # Create all names
            list_col = ["cmd_left_wheel", "cmd_right_wheel"]

            new_dico = create_columns_names_from_dict_with_names(list_col,dico_info,{})
            dico_update.update(new_dico)

            list_df.append(pd.DataFrame.from_dict(dico_update))
    
        df_all_terrain = pd.concat(list_df)


        self.results_file["MRMSE_2d"].update({self.dataset2eval.id:df_all_terrain})
        

        with open(self.path_to_res, 'wb') as file:
                pickle.dump( self.results_file,file)

if __name__=="__main__":

    motion_model_name ="ideal_diff_drive"
    robot_name = "warthog"

    dataset = Dataset2Evaluate("drive_dataset")

    mpe = ModelPredictionErrorComputer(motion_model_name,robot_name,dataset)

    mpe.evaluate_model_on_all_terrains_MRMSE_2d()