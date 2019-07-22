%%%%% this will produce a vector of files per family on same class
function class1 = class1grab(A,family)
    class1vector = A(family,:);
    class1 = class1vector(class1vector ~= "0");

end