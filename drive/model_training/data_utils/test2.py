import yaml 





with open("drive/model_training/data_utils/update_drive_blacklist.yaml") as file:


    blacklist = yaml.safe_load(file)


print("marmotte" in blacklist["robot_to_skip"])