%% data input
data = data_input('two_classes_data');
labels = data_input('two_classes_labels');
ind1 = find(labels == 1);
ind2 = find(labels == 2);
labels(ind1) = -1;
labels(ind2) = 1;
labels = labels(:,1);
labels = labels';
%% feature extraction
feat = [];
% feature includes:
% 1~3: the standard deviation of each time series, each of 3 in (x, y, z)
% 4: the maximum magnitude of data
% 5~6: (concerning the deviation series in a sliding window) the maximum
% values and the sum of the data
% 7~8: (concerning the deviation series in a sliding window) the maximum
% values and the sum of the (row) normalized data, indicating the cosine
% direction
for k = 1:length(data)
    feat_temp = [];
    temp = squeeze(data(k,:,:));
    temp_square = sqrt(temp(:,1).^2 + temp(:,2).^2 + temp(:,3).^2);
    f9 = stdSlideWin(temp, 10);
    temp_norm = normr(temp);
    f10 = stdSlideWin(temp_norm, 10);
    feat_temp = [std(temp), max(temp_square), max(f9), sum(f9),...
        max(f10), sum(f10)];
    feat = [feat; feat_temp];
end
%%
save feat feat
save labels labels



