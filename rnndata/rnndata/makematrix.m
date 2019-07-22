function A = makematrix(img_path)
  img_folder = dir(fullfile(img_path, '*.jpg'));
   gt_nums = numel(img_folder);
   sumw = 0
   sumh = 0
   widA = []
   heiA = []
   A1 = zeros(600,954);
   A = string(A1);
   for i = 1:gt_nums
       img_name = img_folder(i).name;
       img_name_split = strsplit(img_name, '.');
       img_save_name = img_name_split{1};
       for k = 1:600
           j = num2str(k,'%05.f');
           flag = contains(img_save_name, strcat(j,'_'));
           if flag == 1
               S = 0;
               display(img_save_name);
               S = sum(A(k,:)~='0');
               A(k,S+1) = img_save_name;
           else
               continue;
           end
       end
   end 
end

