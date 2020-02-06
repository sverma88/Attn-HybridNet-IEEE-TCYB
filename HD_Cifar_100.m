% ==== HDNet Demo =======
% S. Verma, C. Wang, L. Zhu, W. Liu, 
% "Attn-HybridNet: Improving Deep Learning Networks with two views of Images"

% Sunny verma [Sunny.Verma@student.uts.edu.au]
% Please email me if you find bugs, or have suggestions or questions!
% ========================

parpool('local',4)
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
addpath('./TD_org');
addpath('./tensor_toolbox_2.6');
make;

ImgSize = 32; 
rng_index = 5;
rng(rng_index);

%%%%%% Load the data, download CIFAR-100 dataset and pass its path
load('/home/suverma/Downloads/cifar-100-matlab/train.mat');
TrnData     = data';
TrnLabels   = fine_labels;
clear fine_labels data

load('/home/suverma/Downloads/cifar-100-matlab/test.mat');
TestData    = double(data)';
TestLabels  = fine_labels;
ImgFormat   = 'color'; 

TrnLabels   = double(TrnLabels)  + 1;
TestLabels  = double(TestLabels) + 1;
rndindex    =randperm(50000);
TrnData     = TrnData(:,rndindex);  % sample training samples
TrnLabels   = TrnLabels(rndindex);


nTestImg = length(TestLabels);


%% HDNet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
HDNet.NumStages = 2;
HDNet.PatchSize = [5 5];
HDNet.NumFilters = [27 8];
HDNet.HistBlockSize = [8 8];
HDNet.BlkOverLapRatio = 0.5;
HDNet.Pyramid = [4 2 1];

fprintf('\n ====== HDNet Parameters ======= \n')
HDNet


fprintf('\n ====== HDNet Training ======= \n')
TrnData_ImgCell     = mat2imgcell(double(TrnData),ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
TestData_ImgCell    = mat2imgcell(double(TestData),ImgSize,ImgSize,ImgFormat);

save('Cifar100_labels.mat','TrnLabels','TestLabels');

tic
[V_TD, V_P, ftd, fp, BlkIdx, Tmean] = Hybrid_train(TrnData_ImgCell,HDNet);
toc


fprintf('\n ====== HDNet Test Feature Extraction \n')
test_featP = cell(10000,1);
test_featTD = cell(10000,1);


parfor idx = 1:1:nTestImg    
    [ftest_TD,ftest_P] = Hybrid_FeaExt(TestData_ImgCell(idx),V_TD, V_P, HDNet, Tmean); % extract a test feature using trained HDNet model 
    test_featTD{idx,1} = ftest_TD;
    test_featP{idx,1} = ftest_P;
    idx
    
end

test_featTD = cellfun(@convert_dense,test_featTD,'UniformOutput',false);
test_featTD = [test_featTD{:}]';
test_featP = cellfun(@convert_dense,test_featP,'UniformOutput',false);
test_featP = [test_featP{:}]';
save('Test_TDfeat.mat', 'test_featTD');
save('Test_PCAfeat.mat', 'test_featP');


