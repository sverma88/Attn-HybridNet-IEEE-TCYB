function [Patch2] = extract_patch_ulbp(InTensor, Filters_1, Filters_2, ind, mapping)

% Computing PCA filter outputs
% ======== INPUT ============
% InTensor      A single patch from Input images
% Filter_1      Filters obtained from training layer 1
% Filter_2      Filters obtained from training layer 2
% ind           Label of the cluter Patch belongs to
% ======== OUTPUT ===========
% OutImg           filter output (cell structure)
% OutImgIdx        Image index for OutImg (column vector)
% OutImgIND        Indices of input patches that generate "OutImg"
% ===========================
addpath('./Utils')
addpath('./tensor_Utils')

Size_tensor = size(InTensor);
Patch = tenzeros(2,Size_tensor(1),Size_tensor(2),Size_tensor(3));

Patch(1,:,:,:) = convn(double(InTensor),Filters_1{ind,1},'same');
Patch(2,:,:,:) = convn(double(InTensor),Filters_1{ind,2},'same');
% Patch(3,:,:,:) = convn(double(InTensor),Filters_1{ind,3},'same');

Patch2 = zeros(2,59*2);

Patch2(1,1:59) = lbp(reshape(convn(double(Patch(1,:,:,:)),Filters_2{ind,1},'same'),Size_tensor(1),Size_tensor(2)*Size_tensor(3)),1,8,mapping,'h');
Patch2(1,60:118) = lbp(reshape(convn(double(Patch(1,:,:,:)),Filters_2{ind,2},'same'),Size_tensor(1),Size_tensor(2)*Size_tensor(3)),1,8,mapping,'h');
% Patch2(1,119:end) = lbp(reshape(convn(double(Patch(1,:,:,:)),Filters_2{ind,3},'same'),Size_tensor(1),Size_tensor(2)*Size_tensor(3)),1,8,mapping,'h');

Patch2(2,1:59) = lbp(reshape(convn(double(Patch(2,:,:,:)),Filters_2{ind,1},'same'),Size_tensor(1),Size_tensor(2)*Size_tensor(3)),1,8,mapping,'h');
Patch2(2,60:118) = lbp(reshape(convn(double(Patch(2,:,:,:)),Filters_2{ind,2},'same'),Size_tensor(1),Size_tensor(2)*Size_tensor(3)),1,8,mapping,'h');
% Patch2(2,119:end) = lbp(reshape(convn(double(Patch(2,:,:,:)),Filters_2{ind,3},'same'),Size_tensor(1),Size_tensor(2)*Size_tensor(3)),1,8,mapping,'h');

% Patch2(3,1:59) = lbp(reshape(convn(double(Patch(3,:,:,:)),Filters_2{ind,1},'same'),Size_tensor(1),Size_tensor(2)*Size_tensor(3)),1,8,mapping,'h');
% Patch2(3,60:118) = lbp(reshape(convn(double(Patch(3,:,:,:)),Filters_2{ind,2},'same'),Size_tensor(1),Size_tensor(2)*Size_tensor(3)),1,8,mapping,'h');
% Patch2(3,119:end) = lbp(reshape(convn(double(Patch(3,:,:,:)),Filters_2{ind,3},'same'),Size_tensor(1),Size_tensor(2)*Size_tensor(3)),1,8,mapping,'h');

end






