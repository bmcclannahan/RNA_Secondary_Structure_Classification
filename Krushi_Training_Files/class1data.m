
%%%% this script will genrate the class1 family


path = "/Users/b523m844/RNA_Secondary_Structure_Classification/Test_Data_Set/"
filematrix = makematrix("/Users/b523m844/RNA_Secondary_Structure_Classification/Small_Data_Set/");
number_of_family = size(filematrix,1);
for i = 1:100 %%%%% i is for the family number, i = 1 if for 1st family, 2 is for 2nd,,,,
       family_files = class1grab(filematrix,i);
       combination_from_onefamily = comb(family_files);
       number_of_combination = size(combination_from_onefamily,1);
       for k = 1: number_of_combination
           imgname1 = combination_from_onefamily(k,1);
           imgname2 = combination_from_onefamily(k,2);
           img1_filename = path + imgname1 + ".jpg";
           img2_filename = path + imgname2 + ".jpg";
           I1 = imread(img1_filename);
           I2 = imread(img2_filename);
           mergeimg = mergeImage(I1, I2);
           img_save_path = savename(imgname1, imgname2);
           display(img_save_path);
           imwrite(mergeimg,img_save_path);
        
       end
           
end