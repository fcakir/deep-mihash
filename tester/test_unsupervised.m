function test_unsupervised(net, imdb, batchFunc, opts, metrics, ...
        noLossLayer, subset)
% Test function for unsupervised datasets. 
%
% Please cite the below papers if you use this code.
%
% 1. "Hashing with Mutual Information", 
%    Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
%    arXiv:1803.00974 2018
%
% 2. "MIHash: Online Hashing with Mutual Information", 
%    Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
%    International Conference on Computer Vision (ICCV) 2017
%    (* equal contribution)
%
% INPUTS
%   net 	    - (struct) The neural net. Typically contains 'layers' field and 
% 			      other related information. 
%   imdb        - (struct) Dataset. See get_imdb.m and functions therein.
%   batchFunc   - (function handle) batch sampling function.
%   opts        - (struct) options, see get_opt.m and process_opts.m . 
%   metrics 	- (string) in {'AP', 'AP@5000', 'AP@50000'}. Evaluation metric.
%   noLossLayer - (bool) manages which layer output to get in cnn_encode* functions 
%   subset      - (2D vector) Sample sizes for training and testing sets.
% 				  For evaluation on a subset of the training and testing data. 
%
assert(~isempty(metrics));
if ~iscell(metrics)
    assert(isstr(metrics));
    metrics = {metrics};
end
if ~exist('noLossLayer', 'var')
    noLossLayer = false;
end
if noLossLayer,  layerOffset = 0;
else, layerOffset = 1;
end
myLogInfo(opts.methodID);
myLogInfo(opts.identifier);
assert(opts.unsupervised);

train_id = find(imdb.images.set == 1 | imdb.images.set == 2);
test_id  = find(imdb.images.set == 3);
if exist('subset', 'var')
    train_id = train_id(randperm(numel(train_id), subset(2)));
    test_id = test_id(randperm(numel(test_id), subset(1)));
end
Ntrain = numel(train_id);
Ntest  = numel(test_id);

batch_size = opts.batchSize;
onGPU = ~isempty(opts.gpus);

% -----------------------------------------------------------------------------
% Compute the hash table for query (test) set
% -----------------------------------------------------------------------------
fprintf('Getting (Htest, Xtest)...'); tic;
Htest = zeros(opts.nbits, Ntest, 'single');
Xtest = [];
for t = 1:batch_size:Ntest
    ed = min(t+batch_size-1, Ntest);
    [rex, data] = cnn_encode_unsup(net, batchFunc, imdb, test_id(t:ed), ...
        onGPU, layerOffset);
    Htest(:, t:ed) = single(rex > 0);
    Xtest(t:ed, :) = squeeze(data)';
end
toc;

% -----------------------------------------------------------------------------
% Compute the hash table for retrieval (training) set
% Also, compute the affinity (neighborhood) matrix
% -----------------------------------------------------------------------------
fprintf('Getting (Htrain, Aff)...'); tic;
Htrain = zeros(opts.nbits, Ntrain, 'single');
Aff_bin = zeros(Ntest, Ntrain, 'single');
for t = 1:batch_size:Ntrain
    ed = min(t+batch_size-1, Ntrain);
    [rex, data] = cnn_encode_unsup(net, batchFunc, imdb, train_id(t:ed), ...
        onGPU, layerOffset);
    data = squeeze(data)';
    Htrain(:, t:ed)  = single(rex > 0);
    Aff_bin(:, t:ed) = affinity_binary([], [], Xtest, data, opts);
end
toc;
whos Htest Htrain

% -----------------------------------------------------------------------------
% evaluate
% -----------------------------------------------------------------------------
myLogInfo('Evaluating...');
for m = metrics
    Aff = Aff_bin;
    if ~isempty(strfind(m{1}, '@'))
        s = strsplit(m{1}, '@');
        assert(numel(s) == 2);
        cutoff = str2num(s{2});
        evalFn = str2func(['evaluate_' s{1}]);
    else
        cutoff = [];
        evalFn = str2func(['evaluate_' m{1}]);
    end
    evalFn(Htest, Htrain, Aff, opts, cutoff);
end
end

