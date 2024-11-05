import numpy as np 
import pandas as pd 
import pathlib 
import yaml
from extractors import *
from drive.model_training.models.kinematic.ideal_diff_drive import Ideal_diff_drive
import pickle 
import matplotlib.pyplot as plt 
global PATH_TO_METRIC 
global PATH_TO_SAVE_FOLDER
from datetime import datetime



PATH_TO_METRIC = pathlib.Path('drive/model_training/data_utils/metric_config.yaml')
PATH_TO_SAVE_FOLDER = pathlib.Path('drive_datasets/results_multiple_terrain_dataframe/metric')
RESULTS_FILE_NAME = "metric_results.pkl"

PATH_TO_RESULT_FILE = PATH_TO_SAVE_FOLDER/RESULTS_FILE_NAME

if not PATH_TO_SAVE_FOLDER.is_dir():
    PATH_TO_SAVE_FOLDER.mkdir()




class Dataset2Evaluate():

    def __init__(self,dataset_name) -> None:
        
        with open(PATH_TO_METRIC, 'r') as file:
            config_file = yaml.safe_load(file)
        
        

        for datasets,datasets_info in config_file["datasets"].items():

            if datasets == dataset_name:

                self.datasets_info = datasets_info

        
        path = self.datasets_info["path_dataset"]

        # Extract the dataframe and filter
        self.df = pd.read_pickle(path)
        for column_filter, value in self.datasets_info["filter_columns_and_values"].items():
            self.df = self.df.loc[self.df[column_filter]==value]

        self.id = dataset_name + "_"+ self.datasets_info["filter_columns_and_values"]["robot"] +"_"+ self.datasets_info["filter_columns_and_values"]["traction"] 
        self.terrains = list(self.df.terrain.unique())
        
        self.rate = self.datasets_info["rate"]
    def __getitem__(self, terrain):
        # Extract the columns of the dataframe
        dict_2_update = {"format":self.datasets_info["format"]}

        df_by_terrain = self.df.loc[self.df.terrain == terrain]
        
        for attribute_name, column_name in self.datasets_info["columns_names"].items():
            dict_2_update[attribute_name] = column_type_extractor(df_by_terrain,column_name)
        

        col_2_add = {"dataset_id":self.id}
        col_2_add.update(self.datasets_info["filter_columns_and_values"])
        dict_2_update["col_2_multiply_and_add"] = col_2_add 

        return dict_2_update

    

class DifficultyMetric():


    def __init__(self,metric_name) -> None:
        
        

        with open(PATH_TO_METRIC, 'r') as file:
            metric_param_config = yaml.safe_load(file)
        
        

        
        for metric,params_metric in metric_param_config["metric"].items():

            if metric == metric_name:
                
                self.metric_parameters = params_metric
        

        path_2_robot = self.metric_parameters["robot_rel_path"]
        with open(path_2_robot, 'r') as file:
            robot_param = yaml.safe_load(file)
            

        self.metric_parameters["robot"] = robot_param["robot"]


        self.metric_name = metric_name
        

        # Read results_file 
        list_possible_metric = ["kinetic_energy","kinetic_energy_wheel_encoder",
                                "kinetic_energy_wheel_encoder_ratio","DiffSpeedProprioExteroEnergy",
                                "KineticEnergyWheelOnly","KineticEnergyICPOnly"]
        if not PATH_TO_RESULT_FILE.is_file():
            empty_dict  = {}
            for metric in list_possible_metric:
                empty_dict[metric] = {}
    
            with open(PATH_TO_RESULT_FILE, 'wb') as file:
                results_file = pickle.dump(empty_dict,file)
        

        with open(PATH_TO_RESULT_FILE, 'rb') as file:
            self.results_file = pickle.load(file)
        

class KineticEnergyMetric(DifficultyMetric):

    def __init__(self,metric_name,robot_name) -> None:
        super().__init__(metric_name)

        self.robot_name = robot_name
        # Extract robot params
        for robot,robot_param in self.metric_parameters["robot"].items():
            if robot == self.robot_name:
                self.width = robot_param["width"]
                self.length = robot_param["length"]
                self.basewidth = robot_param["basewidth"]
                self.wheel_radius = robot_param["wheel_radius"]
                self.masse = robot_param["masse"]
        self.inertia_constraints = (self.width**2 + self.length**2)/12
        self.metric_name = "kinetic_energy"

        
        
    def compute_energy(self,vx,vy,omega_body):
        """_summary_

        Args:
            vx (array): assuming that the vector is N by 1
            vy (_type_): assuming that the vector is N by 1
            omega_body (_type_): assuming that the vector is N by 1
        """

        translation_energy = 1/2 * (vx**2+vy**2)
        rotationnal_energy = 1/2 * (self.inertia_constraints * omega_body**2)
        state_kin_energy =  translation_energy + rotationnal_energy

        
        return state_kin_energy,rotationnal_energy, translation_energy

    def compute_kinetic_energy_metric(self,dataset):

        gt_energies = self.compute_energy(dataset["gt_body_lin_vel"],
                                                dataset["gt_body_y_vel"],
                                                dataset["gt_body_yaw_vel"])
        
        # Assuming instant acceleration
        y_cmd =np.zeros(dataset["gt_body_lin_vel"].shape)
        idd_energies = self.compute_energy(dataset["cmd_body_lin_vel"],
                                                y_cmd,
                                                dataset["cmd_body_yaw_vel"])

        if dataset["format"] == "n_cmd x horizon":
            resulting_energy = {}
            energy_order = ["total_energy_metric","rotationnal_energy_metric","translationnal_energy_metric"]
            
            for energy_name, gt_energy, idd_energy in zip(energy_order,gt_energies,idd_energies):

                kinetic_metric = (idd_energy[:,:-1] - gt_energy[:,1:])  / idd_energy[:,:-1] 
                
                resulting_energy[energy_name] = kinetic_metric

        return resulting_energy
    
    def compute_all_terrain(self,dataset,multiple_terrain=False):

        
        list_df = []
        for terrain in dataset.terrains:
            
            dico_data = dataset[terrain]
            shape = dico_data["cmd_left_wheel"].shape
            result_terrain = self.compute_kinetic_energy_metric(dico_data)
        
            result_terrain["cmd_left_wheel"] = dico_data["cmd_left_wheel"]
            result_terrain["cmd_right_wheel"] = dico_data["cmd_right_wheel"]
            # Create all names
            
            new_dict = {}
            list_col = list(result_terrain.keys()) 
            results_dict_1_name_by_column = create_columns_names_from_dict_with_names(list_col,result_terrain,new_dict)
            # Add identifying columns 
            for key,value in dico_data["col_2_multiply_and_add"].items():
                results_dict_1_name_by_column[key] = [value] * shape[0]
            
            results_dict_1_name_by_column["terrain"] = [terrain] * shape[0]
            list_df.append(pd.DataFrame.from_dict(results_dict_1_name_by_column))

        df_all_terrain = pd.concat(list_df)

        # Get the current date and time
        self.results_file[self.metric_name] = {dataset.id:df_all_terrain}

        with open(PATH_TO_RESULT_FILE, 'wb') as file:
            results_file = pickle.dump(self.results_file,file)
        

        
        #self.saving_path = path_to_dataset_folder
        

class KineticEnergyMetricWheelEncoder(KineticEnergyMetric):

    def __init__(self,metric_name,robot_name) -> None:
        super().__init__(metric_name,robot_name)

        
        
        self.metric_name = "kinetic_energy_wheel_encoder"
        self.jacobian = self.wheel_radius *np.array([[1/2, 1/2], [-1/self.basewidth,1/self.basewidth]])
        # Initiate IDD
        # Initiate IDD
        
    def compute_energy_from_wheel_encoder(self,left_wheel_encoder,right_wheel_encoder,vy_array):
        """Compute the energy of equivalent from the wheel encoder motion predicted by the IDD
        Args:
            left_wheel_encoder (_type_): _description_
            right_wheel_encoder (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Compute the vx, vy equivalent speed encoder 

        list_state_kin_energy = [] 
        list_rotationnal_energy = []
        list_translationnal_energy = []
        for horizon in range(left_wheel_encoder.shape[0]):

            u_wheel = np.array([left_wheel_encoder[horizon,:],right_wheel_encoder[horizon,:]])
            body_cmd = self.jacobian @ u_wheel

            vx = body_cmd[0,:]
            vomega = body_cmd[1,:]
            vy = np.zeros(vx.shape) # vy_array[horizon,:]  

            state_kin_energy,rotationnal_energy, translation_energy = self.compute_energy(vx,vy,vomega)

            list_state_kin_energy.append(state_kin_energy)
            list_rotationnal_energy.append(rotationnal_energy)
            list_translationnal_energy.append(translation_energy)

        return np.array(list_state_kin_energy),np.array(list_rotationnal_energy), np.array(list_translationnal_energy)

    
    def compute_kinetic_energy_metric(self,dataset):

        gt_energies = self.compute_energy(dataset["gt_body_lin_vel"],
                                                dataset["gt_body_y_vel"],
                                                dataset["gt_body_yaw_vel"])
        
        # Computing energy from proprioceptive sensors (wheel encoder)
        
        wheel_encoder_energies = self.compute_energy_from_wheel_encoder(dataset["gt_left_wheel"],
                                                dataset["gt_right_wheel"],dataset["gt_body_y_vel"])

        if dataset["format"] == "n_cmd x horizon":
            resulting_energy = {}
            energy_order = ["total_energy_metric","rotationnal_energy_metric","translationnal_energy_metric"]
            
            for energy_name, gt_energy, wheel_encoder_energy in zip(energy_order,gt_energies,wheel_encoder_energies):

                kinetic_metric = (wheel_encoder_energy - gt_energy)  / wheel_encoder_energy
                
                resulting_energy[energy_name] = kinetic_metric

        return resulting_energy
    
    def compute_all_terrain(self,dataset,multiple_terrain=False):

        new_file =False
        list_df = []
        for terrain in dataset.terrains:
            
            dico_data = dataset[terrain]
            shape = dico_data["cmd_left_wheel"].shape
            result_terrain = self.compute_kinetic_energy_metric(dico_data)
        
            result_terrain["cmd_left_wheel"] = dico_data["cmd_left_wheel"]
            result_terrain["cmd_right_wheel"] = dico_data["cmd_right_wheel"]
            # Create all names
            
            list_col = list(result_terrain.keys()) 
            new_dico = {}
            results_dict_1_name_by_column = create_columns_names_from_dict_with_names(list_col,result_terrain,new_dico)
            # Add identifying columns 
            for key,value in dico_data["col_2_multiply_and_add"].items():
                results_dict_1_name_by_column[key] = [value] * shape[0]
            
            results_dict_1_name_by_column["terrain"] = [terrain] * shape[0]
            list_df.append(pd.DataFrame.from_dict(results_dict_1_name_by_column))

        df_all_terrain = pd.concat(list_df)

        # Get the current date and time
        self.results_file[self.metric_name] = {dataset.id:df_all_terrain}

        with open(PATH_TO_RESULT_FILE, 'wb') as file:
            results_file = pickle.dump(self.results_file,file)
        

        
        #self.saving_path = path_to_dataset_folder
        

class KineticEnergyMetricWheelEncoderRatio(KineticEnergyMetricWheelEncoder):

    
    def __init__(self,metric_name,robot_name) -> None:
        super().__init__(metric_name,robot_name)

        
        
        self.metric_name = "kinetic_energy_wheel_encoder_ratio"
        self.jacobian = self.wheel_radius *np.array([[1/2, 1/2], [-1/self.basewidth,1/self.basewidth]])
        # Initiate IDD
        # Initiate IDD
    def compute_kinetic_energy_metric(self,dataset):
        
        gt_energies = self.compute_energy(dataset["gt_body_lin_vel"],
                                                dataset["gt_body_y_vel"],
                                                dataset["gt_body_yaw_vel"])
        
        # Computing energy from proprioceptive sensors (wheel encoder)
        
        wheel_encoder_energies = self.compute_energy_from_wheel_encoder(dataset["gt_left_wheel"],
                                                dataset["gt_right_wheel"],dataset["gt_body_y_vel"])

        if dataset["format"] == "n_cmd x horizon":
            resulting_energy = {}
            energy_order = ["total_energy_metric","rotationnal_energy_metric","translationnal_energy_metric"]
            
            for energy_name, gt_energy, wheel_encoder_energy in zip(energy_order,gt_energies,wheel_encoder_energies):

                kinetic_metric = gt_energy/wheel_encoder_energy
                
                resulting_energy[energy_name] = kinetic_metric

        return resulting_energy
    
class KineticEnergyWheelOnly(KineticEnergyMetricWheelEncoder):

    
    def __init__(self,metric_name,robot_name) -> None:
        super().__init__(metric_name,robot_name)

        
        
        self.metric_name = "KineticEnergyWheelOnly"
        self.jacobian = self.wheel_radius *np.array([[1/2, 1/2], [-1/self.basewidth,1/self.basewidth]])
        # Initiate IDD
        # Initiate IDD
    def compute_kinetic_energy_metric(self,dataset):
        
        gt_energies = self.compute_energy(dataset["gt_body_lin_vel"],
                                                dataset["gt_body_y_vel"],
                                                dataset["gt_body_yaw_vel"])
        
        # Computing energy from proprioceptive sensors (wheel encoder)
        
        wheel_encoder_energies = self.compute_energy_from_wheel_encoder(dataset["gt_left_wheel"],
                                                dataset["gt_right_wheel"],dataset["gt_body_y_vel"])

        if dataset["format"] == "n_cmd x horizon":
            resulting_energy = {}
            energy_order = ["total_energy_metric","rotationnal_energy_metric","translationnal_energy_metric"]
            
            for energy_name, wheel_encoder_energy in zip(energy_order,wheel_encoder_energies):

                kinetic_metric = wheel_encoder_energy
                
                resulting_energy[energy_name] = kinetic_metric

        return resulting_energy

class KineticEnergyICPOnly(KineticEnergyMetricWheelEncoder):

    
    def __init__(self,metric_name,robot_name) -> None:
        super().__init__(metric_name,robot_name)

        
        
        self.metric_name = "KineticEnergyICPOnly"
        self.jacobian = self.wheel_radius *np.array([[1/2, 1/2], [-1/self.basewidth,1/self.basewidth]])
        # Initiate IDD
        # Initiate IDD
    def compute_kinetic_energy_metric(self,dataset):
        
        gt_energies = self.compute_energy(dataset["gt_body_lin_vel"],
                                                dataset["gt_body_y_vel"],
                                                dataset["gt_body_yaw_vel"])
        
        
        if dataset["format"] == "n_cmd x horizon":
            resulting_energy = {}
            energy_order = ["total_energy_metric","rotationnal_energy_metric","translationnal_energy_metric"]
            
            for energy_name, gt_energy in zip(energy_order,gt_energies):

                kinetic_metric = gt_energy
                
                resulting_energy[energy_name] = kinetic_metric

        return resulting_energy

class DiffSpeedProprioExteroEnergy(KineticEnergyMetricWheelEncoder):

    
    def __init__(self,metric_name,robot_name) -> None:
        super().__init__(metric_name,robot_name)

        
        
        self.metric_name = "DiffSpeedProprioExteroEnergy"
        self.jacobian = self.wheel_radius *np.array([[1/2, 1/2], [-1/self.basewidth,1/self.basewidth]])
        # Initiate IDD
        # Initiate IDD
    
    def compute_diff_proprio_extero_energy(self,left_wheel_encoder,right_wheel_encoder,vy_array,gt_vx,gt_vy,gt_vyaw):
        """Compute the energy of equivalent from the wheel encoder motion predicted by the IDD
        Args:
            left_wheel_encoder (_type_): _description_
            right_wheel_encoder (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Compute the vx, vy equivalent speed encoder 

        list_state_kin_energy = [] 
        list_rotationnal_energy = []
        list_translationnal_energy = []
        for horizon in range(left_wheel_encoder.shape[0]):

            u_wheel = np.array([left_wheel_encoder[horizon,:],right_wheel_encoder[horizon,:]])
            body_cmd = self.jacobian @ u_wheel

            vx = body_cmd[0,:]
            vomega = body_cmd[1,:]
            vy = np.zeros(vx.shape) # vy_array[horizon,:]  

            horizon_gt_vx = gt_vx[horizon,:]
            horizon_gt_vy = gt_vy[horizon,:]
            horizon_gt_vyaw = gt_vyaw[horizon,:]

            delta_vx = vx - horizon_gt_vx
            delta_vy = vy - horizon_gt_vy
            delta_vyaw = vomega - horizon_gt_vyaw
            
            state_kin_energy,rotationnal_energy, translation_energy = self.compute_energy(delta_vx,delta_vy,delta_vyaw)

            list_state_kin_energy.append(state_kin_energy)
            list_rotationnal_energy.append(rotationnal_energy)
            list_translationnal_energy.append(translation_energy)

        return np.array(list_state_kin_energy),np.array(list_rotationnal_energy), np.array(list_translationnal_energy)

    
    def compute_kinetic_energy_metric(self,dataset):

        gt_energies = self.compute_energy(dataset["gt_body_lin_vel"],
                                                dataset["gt_body_y_vel"],
                                                dataset["gt_body_yaw_vel"])
        
        # Computing energy from proprioceptive sensors (wheel encoder)
        wheel_encoder_energies = self.compute_energy_from_wheel_encoder(dataset["gt_left_wheel"],
                                                dataset["gt_right_wheel"],dataset["gt_body_y_vel"])

        diff_energies = self.compute_diff_proprio_extero_energy(dataset["gt_left_wheel"],
                                                dataset["gt_right_wheel"],dataset["gt_body_y_vel"],
                                                dataset["gt_body_lin_vel"],dataset["gt_body_y_vel"],
                                                dataset["gt_body_yaw_vel"])

        if dataset["format"] == "n_cmd x horizon":
            resulting_energy = {}
            energy_order = ["total_energy_metric","rotationnal_energy_metric","translationnal_energy_metric"]
            
            for energy_name, diff_energy, wheel_encoder_energy,gt_energy in zip(energy_order,diff_energies,wheel_encoder_energies,gt_energies):

                kinetic_metric = diff_energy/wheel_encoder_energy
                
                resulting_energy[energy_name] = kinetic_metric

        return resulting_energy
    
    def compute_all_terrain(self,dataset,multiple_terrain=False):

        new_file =False
        list_df = []
        for terrain in dataset.terrains:
            
            dico_data = dataset[terrain]
            shape = dico_data["cmd_left_wheel"].shape
            result_terrain = self.compute_kinetic_energy_metric(dico_data)
        
            result_terrain["cmd_left_wheel"] = dico_data["cmd_left_wheel"]
            result_terrain["cmd_right_wheel"] = dico_data["cmd_right_wheel"]
            # Create all names
            new_dico = {}
            list_col = list(result_terrain.keys()) 
            results_dict_1_name_by_column = create_columns_names_from_dict_with_names(list_col,result_terrain,new_dico)
            # Add identifying columns 
            for key,value in dico_data["col_2_multiply_and_add"].items():
                results_dict_1_name_by_column[key] = [value] * shape[0]
            
            results_dict_1_name_by_column["terrain"] = [terrain] * shape[0]
            list_df.append(pd.DataFrame.from_dict(results_dict_1_name_by_column))

        df_all_terrain = pd.concat(list_df)

        # Get the current date and time
        self.results_file[self.metric_name] = {dataset.id:df_all_terrain}

        with open(PATH_TO_RESULT_FILE, 'wb') as file:
            results_file = pickle.dump(self.results_file,file)
        

        
        #self.saving_path = path_to_dataset_folder


class GraphMetric():

    def __init__(self, dataset,metric) -> None:
        
        self.dataset = dataset
        self.path_to_results =  PATH_TO_RESULT_FILE
        self.metric = metric
        
        with open(self.path_to_results, 'rb') as file:  # Open the file in binary read mode
            self.metric_results = pickle.load(file)

    def graph_metric_boxplot_by_terrain(self,percentage=True):



        ## Ravel results
        data_all = self.metric_results[self.metric.metric_name][dataset.id]
        #print_column_unique_column(data_all)
        data_axs = []
        list_terrain = list(data_all.terrain.unique())
        
        list_metric = ["total_energy_metric","rotationnal_energy_metric","translationnal_energy_metric"]
        nb_submetric =3

        for metric in list_metric: 

            dico_metric = {}
            for terrain in list_terrain:

                df_local = data_all.loc[data_all.terrain == terrain]

                dico_metric[terrain] = column_type_extractor(df_local,metric)
            data_axs.append(dico_metric)

        
        
        i = 0 
        fig, axs = plt.subplots(nb_submetric,1)
        fig.suptitle(f"{self.metric.metric_name}")
        fig.subplots_adjust(hspace=0.4,wspace=0.4)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        for data in data_axs:
        
            ###### PLot one ax
            ax= axs[i]
            
            # Prepare data for boxplot
            
            medians = {terrain: np.median(results) for terrain, results in data.items()}

            sorted_terrains = sorted(medians.items(), key=lambda item: item[1])

            
            # Prepare data for boxplot in sorted order
            sorted_labels = [terrain for terrain, _ in sorted_terrains]
            if percentage:
                sorted_results = [np.ravel(data[terrain])*100 for terrain in sorted_labels]
            else:
                sorted_results = [np.ravel(data[terrain]) for terrain in sorted_labels]
            # Define colors for each box
            
            color_dict = {"asphalt":"lightgrey", "ice":"aliceblue","gravel":"papayawhip","grass":"honeydew","sand":"darkgoldenrod"}
            
            colors = [color_dict[terrain] for terrain in sorted_labels]

            

            # Create box plot
            box = ax.boxplot(sorted_results, labels=sorted_labels, patch_artist=True,showfliers=False)

            # Set colors for each box
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)

            # Set title and labels
            ax.set_title('Box Plot of Results by Terrain (Sorted by Median)')
            ax.set_ylabel(f'{list_metric[i]} [%]')


            
            # Show the plot

            i+=1
        plt.show()

    def graph_correlation(self,prediction_model_error):


        ## Ravel results
        data_all = self.metric_results[self.metric.metric_name][dataset.id]
        #print_column_unique_column(data_all)
        data_axs = []
        list_terrain = list(data_all.terrain.unique())
        
        list_metric = ["total_energy_metric","rotationnal_energy_metric","translationnal_energy_metric"]
        nb_submetric =3


        for metric in list_metric: 

            dico_metric = {}
            for terrain in list_terrain:

                df_local = data_all.loc[data_all.terrain == terrain]

                dico_metric[terrain] = column_type_extractor(df_local,metric)
            data_axs.append(dico_metric)

        ## 
        fig, axs = plt.subplots(2,1)




if __name__ == "__main__":

    dataset = Dataset2Evaluate("drive_dataset")
    
    dm = KineticEnergyMetric("kinetic_energy","warthog")
    path_to_result = dm.compute_all_terrain(dataset)


    graph_metric = GraphMetric(dataset,dm)
    graph_metric.graph_metric_boxplot_by_terrain()

    
    dm2 = KineticEnergyMetricWheelEncoder("kinetic_energy_wheel_encoder","warthog")
    path_to_result2 = dm2.compute_all_terrain(dataset)
    graph_metric2 = GraphMetric(dataset,dm2)
    graph_metric2.graph_metric_boxplot_by_terrain()

    dm3 = KineticEnergyMetricWheelEncoderRatio("kinetic_energy_wheel_encoder_ratio","warthog")
    path_to_result2 = dm3.compute_all_terrain(dataset)
    graph_metric3 = GraphMetric(dataset,dm3)
    graph_metric3.graph_metric_boxplot_by_terrain(percentage=True)


    dm4 = DiffSpeedProprioExteroEnergy("DiffSpeedProprioExteroEnergy","warthog")
    path_to_result2 = dm4.compute_all_terrain(dataset)
    graph_metric3 = GraphMetric(dataset,dm4)
    graph_metric3.graph_metric_boxplot_by_terrain(percentage=True)
    
    
    dm5 = KineticEnergyICPOnly("KineticEnergyICPOnly","warthog")
    path_to_result2 = dm5.compute_all_terrain(dataset)
    graph_metric3 = GraphMetric(dataset,dm5)
    graph_metric3.graph_metric_boxplot_by_terrain(percentage=True)
    
    dm6 = KineticEnergyWheelOnly("KineticEnergyWheelOnly","warthog")
    path_to_result2 = dm6.compute_all_terrain(dataset)
    graph_metric3 = GraphMetric(dataset,dm6)
    graph_metric3.graph_metric_boxplot_by_terrain(percentage=True)
    
