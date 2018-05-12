function test_unsupervised(net, imdb, batchFunc, opts, metrics, ...
        noLossLayer, subset)
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

% get Htest and Xtest
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

% get Htrain and fill in Aff incrementally
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

% evaluate
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

