function test_supervised(net, imdb, batchFunc, opts, metrics, ...
    noLossLayer, subset)
% Test function for supervised datasets. 
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

assert(~isempty(metrics));
if ~iscell(metrics)
    assert(isstr(metrics));
    metrics = {metrics};
end
if ~exist('noLossLayer', 'var')
    noLossLayer = false;
end
myLogInfo(opts.methodID);
myLogInfo(opts.identifier);
opts.unsupervised = false;

train_id = find(imdb.images.set == 1 | imdb.images.set == 2);
test_id  = find(imdb.images.set == 3);
if exist('subset', 'var')
    myLogInfo('Sampling random subset: %d test, %d database', subset(1), subset(2));
    test_id = test_id(randperm(numel(test_id), subset(1)));
    train_id = train_id(randperm(numel(train_id), subset(2)));
end

Ytrain   = imdb.images.labels(:, train_id)';
Ytest    = imdb.images.labels(:, test_id)';
whos Ytest Ytrain

% -----------------------------------------------------------------------------
% hash tables
% -----------------------------------------------------------------------------
Htest  = cnn_encode_sup(net, batchFunc, imdb, test_id , opts, noLossLayer);
Htrain = cnn_encode_sup(net, batchFunc, imdb, train_id, opts, noLossLayer);

% -----------------------------------------------------------------------------
% evaluate
% -----------------------------------------------------------------------------
myLogInfo('Evaluating...');
for m = metrics
    % available metics: AP, AP@N
    Aff = affinity_binary(Ytest, Ytrain, [], [], opts);
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

