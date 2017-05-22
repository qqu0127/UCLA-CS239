function [ret] = stdSlideWin(vec, winSize)
% INPUT: vec --     the data vector, [width, 3]
%        winSize -- the sliding window size
% OUTPUT: vector, [width - winSize, 1]

% Compute the std of the samples within a sliding window, in 3 dimensions
% Then compute the magnitude accross 3 dimensions

ret = [];
for i = (winSize + 1):length(vec)
    windows = vec((i - winSize):i, :);
    ret = [ret; std(windows)];
end
ret = sqrt(ret(:,3).^2 + ret(:,2).^2 + ret(:,1).^2);