clc;
close all;
clear all;

file = '/media/bkkvh8/Elements/paras/deepEnhancerv2/Raw_image/';
dirData = dir('/media/bkkvh8/Elements/paras/deepEnhancerv2/Raw_image/*.mat');
rmse = 0
for k = 1:numel(dirData)
    F = fullfile(file,dirData(k).name);
    load(F);
    n
    %rmse = rmse+sqrt(immse(gt_img, out_img))
    %imshow(I)
    %S(k).data = I; % optional, save data.
end

avg_rmse = rmse/numel(dirData)

