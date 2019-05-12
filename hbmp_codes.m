function [imdb, opts] = hbmp_codes(imdb, opts)
tic;
metrics = opts.metrics;
if ~iscell(metrics)
	assert(isstr(metrics));
	metrics = {metrics};
end
if ~isempty(imdb.images.labels)
	% Get unique labels
	if isvector(imdb.images.labels)
		[ulabels, ui, uj] = unique(imdb.images.labels);
		ulabels = ulabels';
	else
		[ulabels, ui, uj] = unique(imdb.images.labels', 'rows');
	end	

	if ~isempty(strfind(metrics{1},'AP'))
		S = affinity_binary(ulabels, ulabels, [], [],opts);
	elseif ~isempty(strfind(metrics{1}, 'NDCG'))
		S = affinity_multlv(ulabels, ulabels, [], [], opts);
	else
		error('Unsupported metric for HBMP');
	end

	% construct hbmp codes
	[GCodes, bit_weights, residual] = binary_matrix_pursuit(S, opts);

	clear S
	% store original labels
	imdb.images.orig_labels = imdb.images.labels;

	% assign the new multidimensional hbmp codes as labels
	imdb.images.labels = (2*single(GCodes(uj,:))-1)';

	% auxillary information
	imdb.images.ulabels     = ulabels;
else
    assert(opts.unsupervised & isfield(imdb.images, 'thr_dist'));
	opts.thr_dist = imdb.images.thr_dist;
    itrain = find(imdb.images.set == 1);
    Xtrain = squeeze(imdb.images.data(:, :, :, itrain))';

    S = affinity_multlv([], [], Xtrain, Xtrain, opts);
	opts.Aff = S; 

	% construct hbmp codes
	[GCodes, bit_weights, residual] = binary_matrix_pursuit(S, opts);

	clear S

	imdb.images.labels = NaN * ones(opts.nbits, length(imdb.images.set));
	imdb.images.labels(:, itrain) = (2*single(GCodes)-1)';
end

imdb.images.GCodes      = GCodes;
imdb.images.bit_weights = bit_weights;
t = toc;
myLogInfo('Binary inference in %g s\n', t);
end

% -----------------------------------------------------------
% initialize generate hbmp codes
% -----------------------------------------------------------
function [GCodes, bit_weights, residual] = binary_matrix_pursuit(S, opts)

S = single(S);

% binary inference
[GCodes, bit_weights, residual] = binary_inference(S, opts.max_iter, opts.tolerance, ...
			opts.weighted, opts.regress, opts.nbits);
			
% return only the top nbits
GCodes = GCodes(:, 1:opts.nbits);

% error checking			
if opts.weighted, assert(~isempty(bit_weights)); else, assert(isempty(bit_weights)); end;
end

