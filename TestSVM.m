% ==== HDNet Demo =======
% S. Verma, C. Wang, L. Zhu, W. Liu,
% "Attn-HybridNet: Improving Deep Learning Networks with two views of Images"

% Sunny verma [Sunny.Verma@student.uts.edu.au]
% Please email me if you find bugs, or have suggestions or questions!
% ========================

clear ;
clc;
addpath(genpath('./Liblinear'));
make;


load('TrnRed_P.mat');
load('TestRed_P.mat');
load('TrnRed_TD.mat')
load('TestRed_TD.mat')
load('Cifar100_labels.mat');



nTestImg = 10000;


%% PCA hashing over histograms
c = [5:5:80];
c = [1 c];

Errors = [];


for iter = 1:numel(c)
    
    fprintf('\n ====== Training Linear SVM Classifier ======= \n')
    display(['now testing c = ' num2str(c(iter)) '...'])
    models = train(TrnLabels, [sparse(TrnRed_TD) sparse(TrnRed_P)], ['-s 1 -c ' num2str(c(iter)) ' -q']); % we use linear SVM classifier (C = 10), calling liblinear library
    
    fprintf('\n ====== HDNet Testing ======= \n')
    
    nCorrRecog = 0;
    RecHistory = zeros(nTestImg,1);
       
    tic
    
    for idx = 1:1:nTestImg
        
        
        [xLabel_est, accuracy, decision_values] = predict(TestLabels(idx),...
            [sparse(TestRed_TD(idx,:)) sparse(TestRed_P(idx,:) )], models, '-q');
        
        if xLabel_est == TestLabels(idx)
            RecHistory(idx) = 1;
            nCorrRecog = nCorrRecog + 1;
        end
        
        if 0==mod(idx,nTestImg/1000)
            fprintf('Accuracy up to %d tests is %.2f%%; taking %.2f secs per testing sample on average. \n',...
                [idx 100*nCorrRecog/idx toc/idx]);
        end
        
    end
    
    Averaged_TimeperTest = toc/nTestImg;
    Accuracy = nCorrRecog/nTestImg;
    ErRate = 1 - Accuracy;
    
    Errors = [Errors; ErRate];
    
    save('Errors1000_HD_SVM.mat','Errors');
    
    
    %% Results display
    fprintf('\n  Testing error rate: %.2f%%', 100*ErRate);
    
end



