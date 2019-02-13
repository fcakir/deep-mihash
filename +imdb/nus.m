function DB = nus(opts, net)
% Construct the imdb mat file from the NUSWIDE dataset. 
% Requires three main files/folders.
% 1. AllLabels81.txt under {sdir} directory (see below). It contains a ~270K x 81
% 	 binary matrix.
% 	 This file is simply the columnwise concatenation of the ~270K length binary 
% 	 membership vectors found in each AllLabels/Labels_{concept}.txt of
% 	 http://dl.nextcenter.org/public/nuswide/Groundtruth.zip . The columns are 
% 	 concatenated in alphabetical order as specified in 
%  	 http://dl.nextcenter.org/public/nuswide/ConceptsList.zip .
% 2. Imagelist.txt under {sdir} directory (see below). 
% 	 Found in http://dl.nextcenter.org/public/nuswide/ImageList.zip .
% 	 Contains the image paths. 
% 3. Actual nuswide images. Should be downloaded and stored in path {sdir} as below. 
% 	 {sdir}/
% 	  |-- AllLabels81.txt
% 	  |-- Imagelist.txt
% 	  |-- Flickr/
% 	  |     |-- actor/
% 	  | 	| 	  |-- 0001_2124494179.jpg
% 	  | 	| 	  |-- 0001_2124494179.jpg
% 	  | 	| 	  |--   ... 
% 	  | 	|-- adminisrative_assistant/
% 	  | 	| 	  |-- 00001_534152430.jpg
% 	  | 	| 	  |--   ... 
% 		...
% The {sdir} directory is under opts.dataDir. opts.dataDir is generally
% set to {opts.localDir}/data . Note that opts.localDir generally is './cachedir'.
% See process_opts.m. 
% The image paths will saved in the DB struct below. Note that this DB struct 
% will be saved on disk in the opts.dataDir directory. See get_imdb.m .  
%
% Keeps images associated with the 21 most frequent concepts. This corresponds to
% a total of number of 195834 images. 
%
% INPUTS
%   opts 	 - (struct) options, see get_opt.m and process_opts.m . 
%   net 	 - (struct) The neural net. Typically contains 'layers' field and 
% 			   other related information. 
%
% OUTPUTS
%   DB (struct)
%       .images (struct)
% 				.data 	(nx1 cell array) Each row is an image path.
% 				.labels (lxn matrix) Each column is the concept membership 
% 						indicator for an image. 
% 				.set    (1xn vector) Each element is from {1,2,3} indicating 
% 						a training, validation and test image, respectively. 
%       .meta (struct)
% 				.sets   (1x3 cell array) corresponds to {'train', 'val', 'test'}. 
% 				
%
sdir = fullfile(opts.dataDir, 'NUSWIDE');

images = textread([sdir '/Imagelist.txt'], '%s');
images = strrep(images, '\', '/');
images = strrep(images, 'C:/ImageData', sdir);

% -----------------------------------------------------------------------------
% get labels
% -----------------------------------------------------------------------------
labels = load([sdir '/AllLabels81.txt']);
myLogInfo('Total images = %g', size(labels, 1));

% -----------------------------------------------------------------------------
% use 21 most frequent labels only
% -----------------------------------------------------------------------------
myLogInfo('Keep 21 most frequent tags, removing rest...');
[val, sel] = sort(sum(labels, 1), 'descend');
labels = labels(:, sel(1:21));
myLogInfo('Min tag freq %d', val(21));

% -----------------------------------------------------------------------------
% remove those without any labels
% -----------------------------------------------------------------------------
keep   = sum(labels, 2) > 0;
labels = labels(keep, :);
images = images(keep);
assert(size(labels, 1) == length(images));
myLogInfo('Keeping # images = %g', sum(keep));

% -----------------------------------------------------------------------------
% split
% -----------------------------------------------------------------------------
set = imdb.split_nus(labels, opts);

% -----------------------------------------------------------------------------
% construct the DB struct
% -----------------------------------------------------------------------------
DB.images.data = images;  % only save image paths, load images during training
DB.images.labels = single(labels)';
DB.images.set = uint8(set)';
DB.meta.sets = {'train', 'val', 'test'} ;
end
