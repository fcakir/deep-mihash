function imdb = labelme_gist(opts, net)
% Construct the imdb mat file from the labelme dataset. 
% We use precomputed GIST descriptors. 
%
% INPUTS
%   opts 	 - (struct) options, see get_opt.m and process_opts.m . 
%   net 	 - (struct) The neural net. Typically contains 'layers' field and 
% 			   other related information. 
%
% OUTPUTS
%   DB (struct)
%       .images (struct)
% 				.data 	  (1x1xmx22019) m corresponds to the feature dim. 
% 					      Typically m=512 if GIST features are used. 
% 				.labels   (empty) This dataset is unsupervised.
% 					 	  indicator for an image. 
%               .thr_dist (vector) distance threshold percentiles. Used to define
% 						  neighborhood. For example, for any two instances, if their 
% 						  l2-norm is smaller than this value, then they're considered neighbors
% 						  with varying degree/level. Note that in the mihash paper
% 						  5% percentile and below level defines the neighborhood.
% 					      See the code below. 
% 				.set      (22019x1 vector) Each element is from {1,2,3} indicating 
% 						  a training, validation or test image, respectively. 
%       .meta (struct)
% 				.sets   (1x3 cell array) corresponds to {'train', 'val', 'test'}. 
basedir = fullfile(opts.dataDir, 'LABELME');
load(fullfile(basedir, 'labelme-gist.mat'), ...
    'gist');
data = single(gist);

% split
ntest = 2000;  ntrain = 5000;
iperm = randperm(size(data, 1));
sets  = ones(size(data, 1), 1);
sets(iperm(1:ntest)) = 3;
sets(iperm(ntest+1:ntest+ntrain)) = 1;
sets(iperm(ntest+ntrain+1:end)) = 2;

% remove mean in any case
Xtrain = data(sets==1, :);
dataMean = mean(Xtrain, 1);
data = bsxfun(@minus, data, dataMean);
if opts.normalize
    % unit-length
    rownorm = sqrt(sum(data.^2, 2));
    data = bsxfun(@rdivide, data, rownorm);
end

% Compute threshold value from several percentiles (hard wired)
N = 5000;
assert(size(Xtrain, 1) >= N);
Xtrain = data(sets==1, :);
thr_dist = prctile(pdist(Xtrain(1:N,:), 'euclidean'), [0.1 0.2 1 5]); 

imdb.images.data = permute(data, [3 4 2 1]);
imdb.images.labels = [];
imdb.images.set = uint8(sets);
imdb.images.thr_dist = single(thr_dist);
imdb.meta.sets = {'train', 'val', 'test'} ;
end
