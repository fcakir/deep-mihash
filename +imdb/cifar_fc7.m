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
% 						Typically m=4096 if corresponds to fc7 layer features of 
% 					    a VGG network. 
% 						image.
% 				.labels (1x60000 vector) Label vector
% 				.set    (1x60000 vector) Each element is from {1,2,3} indicating 
% 						a training, validation or test image. 
%       .meta (struct)
% 				.sets   (1x3 cell array) corresponds to {'train', 'val', 'test'}. 
%
basedir = fullfile(opts.dataDir, 'CIFAR-10');
load(fullfile(basedir,'cifar-vggf.mat')); % trainCNN

data = vggf; 
clear vggf
sets = imdb.split_cifar(labels, opts);

% remove mean in any case
Xtrain = data(sets==1, :);
dataMean = mean(Xtrain, 1);
data = bsxfun(@minus, data, dataMean);

if opts.normalize
    % unit-length
    rownorm = sqrt(sum(data.^2, 2));
    data = bsxfun(@rdivide, data, rownorm);
end

DB.images.data = permute(single(data), [3 4 2 1]);
DB.images.labels = single(labels');
DB.images.set = uint8(sets');
DB.meta.sets = {'train', 'val', 'test'} ;
end
