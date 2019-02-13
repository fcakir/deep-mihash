function DB = cifar(opts, net)
% Construct the imdb mat file from the CIFAR-10 dataset. 
% Will automatically download and construct the DB struct as below. 
% Note that this DB struct will be saved on disk in the opts.dataDir directory. 
% See get_imdb.m . 
%
% INPUTS
%   opts 	 - (struct) options, see get_opt.m and process_opts.m . 
%   net 	 - (struct) The neural net. Typically contains 'layers' field and 
% 			   other related information. 
%
% OUTPUTS
%   DB (struct)
%       .images (struct)
% 				.data 	(32x32x3x60000 tensor) The 4th dim corresponds to the
% 						image index.
% 				.labels (1x60000 vector) Label vector
% 				.set    (1x60000 vector) Each element is from {1,2,3} indicating 
% 						a training, validation and test image, respectively. 
%       .meta (struct)
% 				.sets    (1x3 cell array) corresponds to {'train', 'val', 'test'}. 
%               .classes (10x1 cell array) corresponds to class names for
%                        each labels id 
%       .name  (str)     name of the database 
%       .filepath (str)  file path location for the database file
%

[data, labels, ~, names] = cifar_load_images(opts);
set = imdb.split_cifar(labels, opts);

imgSize = opts.imageSize;
% -----------------------------------------------------------------------------
% NOTE: The following normalization only applies when we're training on 32x32 images
% directly. Do not do any normalization for imagenet pretrained VGG/Alexnet, 
% for which resizing and mean subtraction are done on-the-fly during batch 
% generation.
% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng
% -----------------------------------------------------------------------------
if opts.normalize
    assert(imgSize == 32);

    if opts.contrastNormalization
        z = reshape(data,[],60000) ;
        z = bsxfun(@minus, z, mean(z,1)) ;
        n = std(z,0,1) ;
        z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
        data = reshape(z, imgSize, imgSize, 3, []) ;
    end

    if opts.whitenData
        z = reshape(data,[],60000) ;
        W = z(:,set == 1)*z(:,set == 1)'/60000 ;
        [V,D] = eig(W) ;
        % the scale is selected to approximately preserve the norm of W
        d2 = diag(D) ;
        en = sqrt(mean(d2)) ;
        z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
        data = reshape(z, imgSize, imgSize, 3, []) ;
    end
end

% -----------------------------------------------------------------------------
% construct the output struct
% -----------------------------------------------------------------------------
DB.images.data = data ;
DB.images.labels = labels ;
DB.images.set = uint8(set');
DB.meta.sets = {'train', 'val', 'test'} ;
DB.meta.classes = names.label_names;
end



function [data, labels, set, clNames] = cifar_load_images(opts)
% -----------------------------------------------------------------------------
% Prepare the imdb structure, returns image data with mean image subtracted
% -----------------------------------------------------------------------------
unpackPath = fullfile(opts.dataDir, 'CIFAR-10', 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
    {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);
if any(cellfun(@(fn) ~exist(fn, 'file'), files))
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
    fprintf('downloading %s\n', url) ;
    untar(url, fullfile(opts.dataDir,'CIFAR-10')) ;
end

data   = cell(1, numel(files));
labels = cell(1, numel(files));
sets   = cell(1, numel(files));
for fi = 1:numel(files)
    fd = load(files{fi}) ;
    data{fi} = permute(reshape(fd.data',32,32,3,[]), [2 1 3 4]) ;
    labels{fi} = fd.labels' + 1;  % Index from 1
    sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));
labels = single(cat(2, labels{:})) ;

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

end
