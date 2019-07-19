function final_image = build_image(filename1,filename2)
image = split_image(filename1);
image2 = transpose(split_image(filename2));
final_image = image+image2;
end