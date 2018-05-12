function test_imagenet(net, imdb, batchFunc, opts, metrics, ...
    noLossLayer, subset)

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
disp(imdb.images)
whos Ytest Ytrain

% hash tables
Htest  = cnn_encode_sup(net, batchFunc, imdb, test_id , opts, noLossLayer);
Htrain = cnn_encode_sup(net, batchFunc, imdb, train_id, opts, noLossLayer);

% evaluate
myLogInfo('Evaluating...');
for m = metrics
    % available metics: tieAP, tieNDCG, AP, AP@N, NDCG, NDCG@N
    if ~isempty(strfind(m{1}, 'AP'))
        Aff = affinity_binary(Ytest, Ytrain, [], [], opts);
    else
        Aff = affinity_multlv(Ytest, Ytrain, [], [], opts);
    end
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

