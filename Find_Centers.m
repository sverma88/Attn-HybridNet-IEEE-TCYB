% function [Patch, Patch_Labels, Idx, Centre] = Find_Centers(InImg, Imglbs, PatchSize,K) 
function [Tensor_Patch, Centre] = Find_Centers(InImg, PatchSize,K) 

% =======INPUT=============
% InImg            Input images (cell structure)  
% PatchSize        the patch size, asumed to an odd number.
% Imglbs           Labels of the input images
% K                Number of clusters
% =======OUTPUT============
% Patch            Patches extracted from the images
% Idx              Cluter id of the patch
% Centre           Cluster Centres
% =========================

addpath('./Utils')

% to efficiently cope with the large training samples, if the number of training we randomly subsample 10000 the
% training set to learn PCA filter banks
ImgZ = length(InImg);
MaxSamples = 100000;
NumRSamples = min(ImgZ, MaxSamples); 
RandIdx = randperm(ImgZ);

%% Learning PCA filters (V)
NumChls = size(InImg{1},3);
Rx = [];
% Rl = [];

parfor i = 1:ImgZ
    im = im2col_mean_removal(InImg{i},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    Rx = [Rx;im']; % keep stacking the patches
%     Rl = [Rl; ones(size(im,2),1)*Imglbs(i)];
end

[rowRx, ~] = size(Rx);
rowRx
ind = randperm(rowRx, min(800000,rowRx));

[Idx,Centre] = kmeans(Rx(ind,:),K, 'MaxIter', 5000,'Options',statset('UseParallel',1),'Replicates',8);

[centre_patch,~]=hist(Idx,unique(Idx));

Tensor_Patch = cell(K,1);

parfor i=1:K
    data_idx = find(Idx==i);
    Aux = tenzeros(centre_patch(i),PatchSize,PatchSize,3);
    for j=1:centre_patch(i)
    Aux(j,:,:,1)=reshape(Rx(data_idx(j),1:PatchSize^2),PatchSize,PatchSize);
    Aux(j,:,:,2)=reshape(Rx(data_idx(j),PatchSize^2+1:2*PatchSize^2),PatchSize,PatchSize);
    Aux(j,:,:,3)=reshape(Rx(data_idx(j),2*PatchSize^2+1:end),PatchSize,PatchSize);
    end
    Tensor_Patch{i,1}=Aux;
end

end



 



