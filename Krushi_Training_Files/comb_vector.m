
%%%% take two vector as input produced pair of combination as output %%%%%

function pairs = comb_vector(y1,y2)
     
   [p, q] = meshgrid(y1, y2);
   pairs = [p(:) q(:)]

end