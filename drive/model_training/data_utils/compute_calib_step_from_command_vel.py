import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


def compute_calib_step_from_cmd_vel_x(df,path_2_save,nb_points_by_window=6*20):


    cmd_vel_x = df["cmd_vel_x"].astype(float).to_numpy()
    cmd_vel_yaw = df["cmd_vel_omega"].astype(float).to_numpy()
    
    index =  df["cmd_vel_x"].index
    calib_state = "calib"
    calib_step = 0

    list_array = []


    new_calib_step = ["0"]
    new_calib_state = ["0"]

    calib_step = 0
    i = 1

    list_index = []
    list_cmd = []

    first_step = False
    while i <= (cmd_vel_x.shape[0] - nb_points_by_window ):
        
        
        extract_120_cmd_vel_x = cmd_vel_x[i:nb_points_by_window+i]
        
        extract_120_cmd_vel_yaw = cmd_vel_yaw[i:nb_points_by_window+i]
        

        #print(np.unique(extract_120_cmd_vel_yaw).shape)
        #if (np.unique(extract_120_cmd_vel_x).shape[0] ==  1) or (np.unique(extract_120_cmd_vel_yaw).shape[0] == 1) \
        #    and (extract_120_cmd_vel_yaw[0] != 0.0 or extract_120_cmd_vel_x[0] != 0.0):

        treshold = 10**-5
        
        if ((np.var(extract_120_cmd_vel_x) <=  0.0001) or (np.var(extract_120_cmd_vel_yaw) <=  0.0001)) \
            and (np.abs(extract_120_cmd_vel_yaw[0]) >= treshold or np.abs(extract_120_cmd_vel_x[0]) >= treshold):
            first_step = True    
            
            new_calib_step.extend([f"{calib_step}"]*nb_points_by_window)
            new_calib_state.extend(["calib"]*nb_points_by_window) 
            
            calib_step += 1


            test =2
            list_index.append(i)
            list_cmd.append(calib_state)
            
            
            i += nb_points_by_window

            

        else:
            i += 1 

            if first_step:
                
                new_calib_state.append("idle") 
            else:
                new_calib_state.append("") 
            
            new_calib_step.append(f"{calib_step}")
    
    # Add useless idle to complete the shape 

    shape = df.shape[0]
    nb_2_add = shape - len(new_calib_state)

    new_calib_state.extend(["idle"]*nb_2_add)
    new_calib_step.extend([f"{calib_step}"]*nb_2_add)
    
    # Replace the column 
    new_calib_state = np.array(new_calib_state)
    new_calib_step = np.array(new_calib_step)
    df.calib_state = new_calib_state
    df.calib_step = new_calib_step
    
    df.to_pickle(path_2_save)
    

    test2 = df['calib_state'].to_numpy().astype('str')
    test= df['calib_step'].to_numpy().astype('int')
    print("calib_step", calib_step)
    start = 0
    end = -1

    fig,axs = plt.subplots(2,sharex=True)
    axs[0].plot(np.arange(len(new_calib_step))*0.05,np.array(new_calib_step).astype(float))
    axs[0].vlines(np.array(list_index)*0.05,ymin=-5,ymax=60,color="red")
    axs[0].scatter(index[start:end]*0.05,cmd_vel_x[start:end],s=1.0)
    axs[1].vlines(np.array(list_index)*0.05,ymin=-5,ymax=5,color="red")
    axs[1].scatter(index[start:end]*0.05,cmd_vel_yaw[start:end],s=1.0)
    


if __name__ == "__main__":

    #path_2_dataraw = "drive_datasets/data/warthog/wheels/ice/warthog_wheels_ice/model_training_datasets/raw_dataframe_wronged_calib_0_1.pkl"
    path_2_dataraw = "drive_datasets/data/warthog/wheels/ice/warthog_wheels_ice/model_training_datasets/raw_dataframe_wronged_calib.pkl"
    path_to_save = "/home/nicolassamson/ros2_ws/src/DRIVE/drive_datasets/data/warthog/wheels/ice/warthog_wheels_ice/model_training_datasets/raw_dataframe.pkl"
    df_raw = pd.read_pickle(path_2_dataraw)
    compute_calib_step_from_cmd_vel_x(df_raw,path_to_save)
    print(df_raw.shape)
    

    path_2_dataraw2 = "/home/nicolassamson/ros2_ws/src/DRIVE/drive_datasets/data/warthog/wheels/ice/warthog_wheels_ice_2024_7_30_16h30s21_ice_param/model_training_datasets/raw_dataframe_wronged_calib.pkl"
    path_to_save_2 = "/home/nicolassamson/ros2_ws/src/DRIVE/drive_datasets/data/warthog/wheels/ice/warthog_wheels_ice_2024_7_30_16h30s21_ice_param/model_training_datasets/raw_dataframe.pkl"
    
    df_raw_2 = pd.read_pickle(path_2_dataraw2)
    print(df_raw_2.shape)
    compute_calib_step_from_cmd_vel_x(df_raw_2[:int(1127/0.05)],path_to_save_2)


   # path_to_example = "/home/nicolassamson/ros2_ws/src/DRIVE/drive_datasets/data/warthog/wheels/ice/warthog_wheels_ice_ral2023/model_training_datasets/raw_dataframe.pkl"
#
   # df_ex = pd.read_pickle(path_to_example)
#
   # df_ex.calib_step = df_ex.calib_step.astype(float).to_numpy()
#
   # calib_value = df_ex.loc[df_ex.calib_state == "calib"]
   # 
   # 
   # #df_ex.to_csv("/home/nicolassamson/ros2_ws/src/DRIVE/drive_datasets/data/warthog/wheels/ice/warthog_wheels_ice_2024_7_30_16h30s21_ice_param/model_training_datasets/test.csv")
    #df_raw.to_csv("/home/nicolassamson/ros2_ws/src/DRIVE/drive_datasets/data/warthog/wheels/ice/warthog_wheels_ice_2024_7_30_16h30s21_ice_param/model_training_datasets/test_raw.csv")
    #fig,axs = plt.subplots(2,sharex=True)
    #
    #plt.show()
#
#
    #path_2_dataraw3 = "/home/nicolassamson/ros2_ws/src/DRIVE/drive_datasets/data/warthog/wheels/asphalt/warthog_wheels_asphalt_2024_9_20_8h21s48/model_training_datasets/raw_dataframe_without_calib_step.pkl"
    #path_to_save_3 = "/home/nicolassamson/ros2_ws/src/DRIVE/drive_datasets/data/warthog/wheels/asphalt/warthog_wheels_asphalt_2024_9_20_8h21s48/model_training_datasets/raw_dataframe.pkl"
    #
    #df_raw_3 = pd.read_pickle(path_2_dataraw3)
    #print(df_raw_3.shape)
    #compute_calib_step_from_cmd_vel_x(df_raw_3,path_to_save_3)
    #plt.show()
#
#
