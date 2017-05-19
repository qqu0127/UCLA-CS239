function [ret] = data_input(path)
% Author:   Qi Qu
% INPUT: path of .mat data
% OUTPUT: 3D matrix, [#samples, 3, signalWidth]

%load from .mat file, get access to the name of the data field 

raw_data = load(path);
name = fieldnames(raw_data);
name = name{1};
raw_data = raw_data.(name);
ret = [];

for k = 1:3
    index = k:3:size(raw_data, 1);
    ret(:,:,k) = raw_data(index, :);
end

end