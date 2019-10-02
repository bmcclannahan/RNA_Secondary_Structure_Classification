function build_data_set(folder_location,prefix)
    map = build_image_map(folder_location);
    num_keys = length(map.keys)
    keys = map.keys;
    for k = 1:num_keys
        key = cell2mat(keys(k));
        family = key(1);
        family_arr = map(family);
        for i = 1:length(family_arr)
            %generate all same family images
            for j = 1:length(family_arr)
                if i ~= j
                    save_image(folder_location,family_arr(i), family_arr(j), prefix);
                end
            end
            %generate all different family images
            for j = 1:num_keys
                if k ~= j
                    key2 = cell2mat(keys(j));
                    family2 = key2(1);
                    family2_arr = map(family2);
                    for n = 1:length(family2_arr)
                        save_image(folder_location, family_arr(i), family2_arr(n), prefix);
                    end
                end
            end
        end
    end
end