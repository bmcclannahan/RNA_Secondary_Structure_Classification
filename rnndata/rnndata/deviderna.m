img_path = "/Users/krushipatel/Desktop/RNA/val/class2/"
des_path = "/Users/krushipatel/Desktop/RNAnew/test/class2/"
img_folder = dir(fullfile(img_path, '*.jpg'));
gt_nums = numel(img_folder)
for i = 1:3:gt_nums
   filename = img_folder(i).name
   src = fullfile(img_path, filename)
   des = fullfile(des_path, filename)
   movefile(src,des)
   
    
    
    
end