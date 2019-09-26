function A = bppmToJPEG(dirName)

%dirName directory from which .bppm files have to be read and respective .jpg files will be stored...
%     if 7==exist(dirName,'dir')

     %Files=getAllFiles(dirName, '.bppm', true);%%dir(dirName,'*.bppm');
        files = dir(dirName + '*.bppm');

        L = length(files);

        for i=1:L

        file=files(i).name;

        filepath = fullfile( dirName, file );

        [pathstr, name, ext] = fileparts(filepath);

        A = dlmread(filepath);
        normA = A - min(A(:));
        normA = (normA ./ max(normA(:)));
        imwrite(normA,[pathstr,'\',name,'.jpg']);

        end

%     else
%         error('Error: No .bppm file found at given directory. Please try again...');
%     end
end