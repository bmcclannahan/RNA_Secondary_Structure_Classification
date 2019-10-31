function image = split_image(file_name)
    image = imread(file_name);
    image = imresize(image,[224,224]);
    for i = 1:112
        for j = i:112
            if i == j
                image(i,j) = 1;
            else
                image(i,j) = 0;
            end
        end
    end
end