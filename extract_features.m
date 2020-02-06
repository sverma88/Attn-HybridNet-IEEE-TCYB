function [Bow, label] = extract_features(Data, Cluster_centres, BOW_Centres, PatchSize, Imglbs, Filters_L1, Filters_L2)
% =======INPUT=============
% Data             Input images (cell structure)  
% PatchSize        the patch size, asumed to an odd number.
% Imglbs           Labels of the input images
% Trained_V        Factor matrices learned from training
% BOW_Centres      Bow Cluster Centre
% Cluster_centres  Initial Cluster Centre
% K                Number of Centres
% =======OUTPUT============
% features         BOW histogram
% label            label of the image
% =========================

mapping = getmapping(8,'u2');

Number_Images = size(Data,1);

label = Imglbs;

Bow = [];

parfor i=1:Number_Images
    im = im2col_mean_removal(Data{i},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    distances = pdist2(im',Cluster_centres);
    [~,ind]=sort(distances,2);
    tensor_im  = convert_to_tensor(im', PatchSize);
    num_patches = size(tensor_im,1);
    ulbp_images = [];
    
   for j=1:num_patches
        ulbp_images =[ulbp_images; extract_patch_ulbp(tensor_im(j,:,:,:),Filters_L1, Filters_L2, ind(j,1), mapping)];
        
   end
    
    distances = pdist2(ulbp_images,BOW_Centres);
    [~,ind]=sort(distances,2);
    
    Bow = [Bow; histc(ind(:,1),[1:size(BOW_Centres,1)])'];
    
end


end
    
    