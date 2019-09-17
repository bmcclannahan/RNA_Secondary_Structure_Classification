function build_data_set(folder_location,num_epochs,prefix)
map = build_image_map(folder_location);
num_keys = length(map.keys)
keys = map.keys;
for e = 1:num_epochs
    rn = uint16(rand()*(num_keys-1)+1);
    key = cell2mat(keys(rn));
    family = key(1);
    rn = uint16(rand()*(length(map(family))-1)+1);
    family_arr = map(family);
    file1 = family_arr(rn);
    previous = [rn,0,0,0,0];
    for i = 2:5
        rn = uint16(rand()*length(map(family)));
        while ismember(rn,previous)
            rn = uint16(rand()*(length(map(family))-1)+1);
        end
        previous(i) = rn;
        save_image(folder_location,file1, family_arr(rn), prefix);
    end
    for i = 1:4
        rn = uint16(rand()*(num_keys-1)+1);
        while rn == family
            rn = uint16(rand()*(num_keys-1)+1);
        end
        key2 = cell2mat(keys(rn));
        family2 = key2;
        rn = uint16(rand()*(length(map(family2))-1)+1);
        family_arr2 = map(family2);
        save_image(folder_location,file1, family_arr2(rn), prefix);
    end
end
end