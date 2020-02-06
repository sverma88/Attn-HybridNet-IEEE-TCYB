function [BOW_features, Img_labels] = extract_features_revised(InImg, InImgIdx, BOW_Centres, V, PCANet)
% =======INPUT=============
% InImg            Input images (cell structure)  
% PCANet           PCANet stucture details
% InImgIdx         Labels of the input images
% Trained_V        Factor matrices learned from training
% BOW_Centres      Bow Cluster Centre
% =======OUTPUT============
% features         BOW histogram
% label            label of the image
% =========================

mapping = getmapping(8,'u2');


%%%%%%%%%%%% Extracting Features for Layer 1 %%%%%%%%%%%
stage=1;
[OutImg OutImgIdx] = Tensor_output_revised(InImg, InImgIdx, ...
PCANet.PatchSize(stage), V{stage});  


%%%%%%%%%%%%% Doing For Layer 2 and extracting BOW Features %%%%%%%%%%%

stage=2;
[OutImg OutImgIdx] = Tensor_output_revised(OutImg, OutImgIdx, ...
PCANet.PatchSize(stage), V{stage});  

PatchSize = PCANet.PatchSize(stage);

BOW_features = [];
Img_labels = [];

for i = 1:6:length(OutImgIdx)
    im1 = im2colstep(OutImg{i},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    im2 = im2colstep(OutImg{i+1},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    im3 = im2colstep(OutImg{i+2},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    im4 = im2colstep(OutImg{i+3},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    im5 = im2colstep(OutImg{i+4},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    im6 = im2colstep(OutImg{i+5},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    
    im1 = cell2mat(arrayfun(@(n) complbp(im1(:,n),5, mapping), 1:size(im1,2), 'UniformOutput', false)');
    im2 = cell2mat(arrayfun(@(n) complbp(im2(:,n),5, mapping), 1:size(im2,2), 'UniformOutput', false)');
    im3 = cell2mat(arrayfun(@(n) complbp(im3(:,n),5, mapping), 1:size(im3,2), 'UniformOutput', false)');
    im4 = cell2mat(arrayfun(@(n) complbp(im4(:,n),5, mapping), 1:size(im4,2), 'UniformOutput', false)');
    im5 = cell2mat(arrayfun(@(n) complbp(im5(:,n),5, mapping), 1:size(im5,2), 'UniformOutput', false)');
    im6 = cell2mat(arrayfun(@(n) complbp(im6(:,n),5, mapping), 1:size(im6,2), 'UniformOutput', false)');
        
    im = im1 + im2 + im3 + im4 + im5 +im6;
    
    
    distances = pdist2(im,BOW_Centres);
    [~,ind]=sort(distances,2);
    
    BOW_features = [BOW_features; histc(ind(:,1),[1:size(BOW_Centres,1)])'];
    Img_labels = [Img_labels; OutImgIdx(i)];
    
end



end
    
    