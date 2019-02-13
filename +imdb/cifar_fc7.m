function DB = cifar_fc7(opts, net)
% Construct the imdb mat file from the CIFAR-10 dataset. 
% Currently the below files contains VGGF fc7 layer features. 
% 
% INPUTS
%   opts 	 - (struct) options, see get_opt.m and process_opts.m . 
%   net 	 - (struct) The neural net. Typically contains 'layers' field and 
% 			   other related information. 
%
% OUTPUTS
%   DB (struct)
%       .images (struct)
% 				.data 	(1x1xmx60000 tensor) m corresponds to the feature dim.
% 						Typically m=4096 if features correspond to the fc7 layer
% 					    output of a VGG model. 
% 						image.
% 				.labels (1x60000 vector) Label vector
% 				.set    (1x60000 vector) Each element is from {1,2,3} indicating 
% 						a training, validation and test image, respectively. 
%       .meta (struct)
% 				.sets   (1x3 cell array) corresponds to {'train', 'val', 'test'}. 
%
%       .name  (str)     name of the database 
%       .filepath (str)  file path location for the database file
%
basedir = fullfile(opts.dataDir, 'CIFAR-10');
load(fullfile(basedir,'cifar-vggf.mat')); % trainCNN

data = vggf; 
clear vggf
set = imdb.split_cifar(labels, opts);

% -----------------------------------------------------------------------------
% remove mean in any case
% -----------------------------------------------------------------------------
Xtrain = data(set==1, :);
dataMean = mean(Xtrain, 1);
data = bsxfun(@minus, data, dataMean);

% -----------------------------------------------------------------------------
% unit normalize
% -----------------------------------------------------------------------------
if opts.normalize
    rownorm = sqrt(sum(data.^2, 2));
    data = bsxfun(@rdivide, data, rownorm);
end

% -----------------------------------------------------------------------------
%  create the output struct
% -----------------------------------------------------------------------------
DB.images.data = permute(single(data), [3 4 2 1]);
DB.images.labels = single(labels);
DB.images.set = uint8(set');
DB.meta.sets = {'train', 'val', 'test'} ;
end
