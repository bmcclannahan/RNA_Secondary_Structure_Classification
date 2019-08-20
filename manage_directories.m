function manage_directories(data_set_folder,training_set_folder)
p = 'Building Data Set'
build_data_set(data_set_folder,1)
p = 'Dataset built, moving files'
files = dir('*.jpg');
num_images = length(files)
for i = 1:length(files)
    file = files(i).name;
    movefile(file, training_set_folder);
end
p = 'Files moved.'
end