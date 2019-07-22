%%%% take 2 images as input and produce mask for upper and lower image and
%%%% combine both images

function mergeimg = mergeImage(I, J)
    I1 = imresize(I,[150 150]);
    %subplot(2,2,1)
    %imshow(I1)
    J1 = imresize(J,[150 150]);
    %subplot(2,2,2)
    %imshow(J1)
    BW = roipoly(I1,[0 150 150],[0 0 150]);
    BW1 = not(BW);
    Im1 = immultiply(I1,BW);
    %subplot(2,2,3)
    %imshow(Im1)
    Im2 = immultiply(J1,BW1);
    %subplot(2,2,4)
    %imshow(Im2)
    mergeimg = imadd(Im1,Im2);
    %figure;
    %imshow(mergeimg)
    
end
