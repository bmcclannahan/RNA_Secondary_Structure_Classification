train_data_folder = "/home/b523m844/data/Full_Image_Set/Medium_Small_Families/train/";
val_data_folder = "/home/b523m844/data/Full_Image_Set/Medium_Small_Families/val/";
test_data_folder = "/home/b523m844/data/Full_Image_Set/Medium_Small_Families/test/";
%build_data_set(train_data_folder,"train")
build_data_set(val_data_folder,"val")
%build_data_set(test_data_folder,"test")
