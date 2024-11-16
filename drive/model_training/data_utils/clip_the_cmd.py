import pandas as pd 
import numpy as np 
import pathlib
import matplotlib.pyplot as plt


def clipp_to_cmd_max_vel(path, max_vel_lin, max_vel_ang,debug=True):

    path_2_do = pathlib.Path(path)


    df = pd.read_pickle(path_2_do)


    col = df.cmd_vel_omega.astype(float).to_numpy()
    col2 = np.clip(col,-max_vel_ang,max_vel_ang)
    df.cmd_vel_omega = col2


    col3 = df.cmd_vel_x.astype(float).to_numpy()
    col4 = np.clip(col3,-max_vel_lin,max_vel_lin)
    df.cmd_vel_x = col4
    
    
    if debug ==True:
        fig, axs =  plt.subplots(1,1)

        axs.scatter(col,col3)
        axs.scatter(col2,col4)

        axs.vlines([-max_vel_ang,max_vel_ang],-max_vel_lin,max_vel_lin)
        axs.hlines([-max_vel_lin,max_vel_lin],-max_vel_ang,max_vel_ang)
        plt.show()

    path_2_save = path_2_do.parent/"raw_dataframe.pkl"

    df.to_pickle(path_2_save)


def clipped_the_grass_husky_cmd():

    """Bad states had been set to the folowing dataset. 
    """
    path_1 = "drive_datasets/data/husky/wheels/wetgrass/husky_wheels_wetgrass_2024_11_7_9h31s46/model_training_datasets/raw_dataframe_before_clipping.pkl"
    path_2 = "drive_datasets/data/husky/wheels/wetgrass/husky_wheels_wetgrass_2024_11_7_9h56s3/model_training_datasets/raw_dataframe_before_clipping.pkl"
    path_3 = "drive_datasets/data/husky/wheels/wetgrass/husky_wheels_wetgrass_2024_11_7_10h12s52/model_training_datasets/raw_dataframe_before_clipping.pkl"
    path_4 = "drive_datasets/data/husky/wheels/wetgrass/husky_wheels_wetgrass_2024_11_7_11h2s1/model_training_datasets/raw_dataframe_before_clipping.pkl"
    
    
    max_vel_lin = 2.0
    max_vel_ang = 1.0

    for path in [path_1,path_2, path_3, path_4]:
        clipp_to_cmd_max_vel(path, max_vel_lin, max_vel_ang)


if __name__=="__main__":

    clipped_the_grass_husky_cmd()
    