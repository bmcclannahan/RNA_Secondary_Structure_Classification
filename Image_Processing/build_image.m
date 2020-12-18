function final_image = build_image(filename1,filename2)
image = tril(imresize(imread(filename1),[224,224]));
image2 = triu(imresize(imread(filename2),[224,224]));
final_image = image+image2;
end