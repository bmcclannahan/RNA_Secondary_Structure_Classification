function manage_directories(data_set_folder,training_set_folder)
build_data_set(data_set_folder,1000)
files = dir('*.jpg');
for i = 1:length(files)
    file = files(i).name;
    movefile(file, training_set_folder)
end
end