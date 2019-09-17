data_folder = "Small_Data_Set/";
num_epochs = 500
build_data_set(data_folder+"train/",num_epochs,"train")
build_data_set(data_folder+"val/",num_epochs,"val")
build_data_set(data_folder+"test/",num_epochs,"test")