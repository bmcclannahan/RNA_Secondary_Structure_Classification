function map = build_image_map(folder_name)
files = dir(folder_name + "*.jpg")
map = containers.Map(double(1), [1,2]);
remove(map,1);
length(files);
for i = 1:length(files)
    file = "" + files(i).name;
    file_split = split(file,"_");
    family_split = split(file_split(1),"F");
    family = str2double(char(family_split(2)));
    if map.isKey(family)
        family_length = length(map(family));
        family_arr = map(family);
        family_arr(family_length+1) = file;
        map(family) = family_arr;
    else
        map(family) = file;
    end
end
for i = 1:1099
    if map.isKey(i)
        if length(map(i)) < 10
            remove(map,i);
        end
    end
end
end