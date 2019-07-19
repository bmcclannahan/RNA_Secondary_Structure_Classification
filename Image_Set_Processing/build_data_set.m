function build_data_set(folder_name)
files = dir(folder_name+'*.jpg');
files = files([1,230,1000,5000,8000,10000,12000,15000,20000]);
for i = 1:length(files)
    file1 = files(i).name;
    file_split1 = split(file1,"_");
    family_split = split(file_split1(1),'F');
    number_split = split(file_split1(2),'.');
    family1 = str2double(char(family_split(2)));
    number1 = str2double(char(number_split(1)));
    file_location1 = folder_name + file1;
    for j = i:length(files)
        if i ~= j
            file2 = files(j).name;
            file_split2 = split(file2,"_");
            family_split = split(file_split2(1),'F');
            number_split = split(file_split2(2),'.');
            family2 = str2double(char(family_split(2)));
            number2 = str2double(char(number_split(1)));
            file_location2 = folder_name + file2;
            file_suffix = 0;
            if family1 == family2
                file_suffix = 1;
            end
            new_file_name_array = cellstr(["",family1,family2,number1,number2,file_suffix]);
            new_file_name = "C" + strjoin(new_file_name_array,"_") + ".jpg";
            final_image = build_image(file_location1,file_location2);
            imwrite(final_image, new_file_name);
        end
    end
end
end