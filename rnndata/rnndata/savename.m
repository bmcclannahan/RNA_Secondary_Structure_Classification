function save_image_path = savename(filename1, filename2)
    %name_split1 = strsplit(filename1,'.')
     %name_split2 = strsplit(filename2,'.')
     %name_split11 = name_split1{1}
     %name_split12 = name_split2{1}
     fullname = strcat(filename1,filename2,'.jpg');
     save_image_path = "/Users/krushipatel/Desktop/RNATEST/class2/" + fullname;
end