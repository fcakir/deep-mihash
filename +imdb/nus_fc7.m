function DB = nus_fc7(opts, net)
% Construct the imdb mat file from the NUSWIDE dataset. 
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
% 				.data 	(1x1xmxn tensor) m corresponds to the feature dim. 
% 						n corresponds to number of images.
% 						Typically m=4096 corresponding to fc7 layer features of 
% 					    a VGG network. If most frequent 21 concepts has been 
% 						considered then	n=195834. 
% 				.labels (lxn matrix) Each column is the concept membership 
% 						indicator for an image. 
% 				.set    (1xn vector) Each element is from {1,2,3} indicating 
% 						a training, validation and test image, respectively. 
%       .meta (struct)
% 				.sets   (1x3 cell array) corresponds to {'train', 'val', 'test'}. 
%
[data, labels] = load_fc7_nus(opts, true);
set = imdb.split_nus(labels, opts);

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
DB.images.labels = single(labels)';
DB.images.set = uint8(set');
DB.meta.sets = {'train', 'val', 'test'} ;
end


function [X, Y] = load_fc7_nus(opts, use21FrequentConcepts)
if nargin < 1, use21FrequentConcepts = true; end
basedir = fullfile(opts.dataDir, 'NUSWIDE');

load(fullfile(basedir, 'nuswide-vggf.mat'));  % vggf, labels
X = single(vggf); clear vggf
Y = labels;

% use 21 most frequent labels only
if use21FrequentConcepts
    myLogInfo('Using 21 most frequent concepts, removing rest...');
    [~, fi_] = sort(sum(Y, 1), 'descend');
    Y(:, fi_(22:end)) = [];
    fi2_ = find(sum(Y, 2) == 0);
    Y(fi2_, :) = [];
    X(fi2_, :) = [];
    myLogInfo('# points = %g, dim = %g, # labels = %g', ...
        size(X,1), size(X, 2), size(Y, 2));
end
end
