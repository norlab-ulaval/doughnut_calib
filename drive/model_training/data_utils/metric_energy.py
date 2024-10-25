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
        
    def __getitem__(self, terrain):
        # Extract the columns of the dataframe
        dict_2_update = {"format":self.datasets_info["format"]}

        df_by_terrain = self.df.loc[self.df.terrain == terrain]
        
        for attribute_name, column_name in self.datasets_info["columns_names"].items():
            dict_2_update[attribute_name] = column_type_extractor(df_by_terrain,column_name)

        return dict_2_update


class DifficultyMetric():


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
        # Initiate IDD
        
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

        results = {"last_update":datetime.now(),"results_by_terrain":{}} 
        for terrain in dataset.terrains:
                
            results["results_by_terrain"][terrain] = self.compute_kinetic_energy_metric(dataset[terrain])
        
        path_to_metric_folder = PATH_TO_SAVE_FOLDER/self.metric_name
        if not path_to_metric_folder.is_dir():
            path_to_metric_folder.mkdir()
        
        #path_to_dataset_folder = path_to_metric_folder/(dataset.id+".pkl")
        
        # Get the current date and time
        self.results_file[self.metric_name] = {dataset.id:results}

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

    def graph_metric_boxplot_by_terrain(self):



        ## Ravel results
        data_all = self.metric_results[self.metric.metric_name][dataset.id]["results_by_terrain"]

        list_terrain = list(data_all.keys())
        list_submetric = list(data_all[list_terrain[0]].keys())
        nb_submetric = len(list_submetric)
        
        data_axs = [{}]*nb_submetric
        
        for j,submetric in enumerate(list_submetric):

            dict_temp = {}
            for terrain, submetrics in data_all.items():
                dict_temp[terrain] = submetrics[submetric] 
            data_axs[j] = dict_temp
        i = 0 
        fig, axs = plt.subplots(nb_submetric,1)
        for data in data_axs:
        
            ###### PLot one ax
            ax= axs[i]
            
            # Prepare data for boxplot
            
            medians = {terrain: np.median(results) for terrain, results in data.items()}

            sorted_terrains = sorted(medians.items(), key=lambda item: item[1])

            
            # Prepare data for boxplot in sorted order
            sorted_labels = [terrain for terrain, _ in sorted_terrains]
            sorted_results = [np.ravel(data[terrain])*100 for terrain in sorted_labels]


            # Define colors for each box
            
            color_dict = {"asphalt":"lightgrey", "ice":"aliceblue","gravel":"papayawhip","grass":"honeydew","sand":"darkgoldenrod"}
            
            colors = [color_dict[terrain] for terrain in sorted_labels]

            

            # Create box plot
            box = ax.boxplot(sorted_results, labels=sorted_labels, patch_artist=True)

            # Set colors for each box
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)

            # Set title and labels
            ax.set_title('Box Plot of Results by Terrain (Sorted by Median)')
            ax.set_ylabel(f'{list_submetric[i]} [%]')


            ax.set_ylim(-50,150)
            # Show the plot

            i+=1
        plt.show()


if __name__ == "__main__":

    dataset = Dataset2Evaluate("drive_dataset")

    dm = KineticEnergyMetric("kinetic_energy","warthog")

    path_to_result = dm.compute_all_terrain(dataset)


    graph_metric = GraphMetric(dataset,dm)
    graph_metric.graph_metric_boxplot_by_terrain()

        
