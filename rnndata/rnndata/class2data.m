

%%% this will generate a class2 - combination of different family

path = "/Users/krushipatel/Desktop/s1500/"
filematrix = makematrix("/Users/krushipatel/Desktop/s1500/");
number_of_family = size(filematrix,1);
j = 102:115;
family_pair = comb(j)
k = size(family_pair,1)

for i = 1:1:k
    family1 = family_pair(i,1); %%% index of family 1%%%%%
    family2 = family_pair(i,2); %%% index of family 2 %%%%%
    [family_1, family_2] = class2grab(filematrix,family1, family2)
    combination_from_two_family = comb_vector(family_1, family_2)
    number_of_combination = size(combination_from_two_family,1);
    for l = 1:number_of_combination
        imgname1 = combination_from_two_family(l,1)
        imgname2 = combination_from_two_family(l,2)
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