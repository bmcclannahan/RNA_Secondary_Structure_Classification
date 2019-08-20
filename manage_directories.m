function manage_directories(data_set_folder,training_set_folder)
disp('Building Data Set')
build_data_set(data_set_folder,3000)
disp('Dataset built, moving files')
files = dir('*.jpg');
num_images = length(files)
for i = 1:length(files)
    file = files(i).name;
    movefile(file, training_set_folder);
end
disp('Files moved.')
end