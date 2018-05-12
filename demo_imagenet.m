function demo_imagenet(nbits, modelType, varargin)

% init opts
ip = inputParser;
ip.addParameter('split', 1);
ip.addParameter('obj', 'fastap'); % fastap or aplb
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = get_opts(ip.Results, 'imagenet', nbits, modelType, varargin{:});

%%%%%%%%%%%%%%%%% hard-coded fields %%%%%%%%%%%%%%%%%
assert(ismember(opts.modelType, {'alexnet_ft', 'vggf_ft'}));
opts.metrics = {'AP@1000'};
opts.testFunc = @test_imagenet;
%%%%%%%%%%%%%%%%% hard-coded fields %%%%%%%%%%%%%%%%%

% post-parsing
cleanupObj = onCleanup(@cleanup);
opts = process_opts(opts);  % carry out all post-processing on opts
record_diary(opts);
opts
myLogInfo(opts.methodID);
myLogInfo(opts.identifier);

% ---------------
% model & data
% ---------------
if ~isempty(opts.gpus) && opts.gpus == 0
    opts.gpus = auto_select_gpu;
end
[net, opts] = get_model(opts);

% imdb
global imdb
imdb = get_imdb(imdb, opts, net);

% ---------------
% train
% ---------------
sz = [opts.imageSize opts.imageSize];
meanImage = single(net.meta.normalization.averageImage);
if isequal(size(meanImage), [1 1 3])
    meanImage = repmat(meanImage, sz);
else
    assert(isequal(size(meanImage), [sz 3]));
end
testBatchFunc = @(I, B) batch_imagenet(I, B, opts.imageSize, meanImage);
trainBatchFunc = @batch_simplenn;

% figure out learning rate vector
if opts.lrdecay>0 & opts.lrdecay<1
    cur_lr = opts.lr;
    lrvec = [];
    while length(lrvec) < opts.epoch
        lrvec = [lrvec, ones(1, opts.lrstep)*cur_lr];
        cur_lr = cur_lr * opts.lrdecay;
    end
else
    lrvec = opts.lr;
end
saveps = 0: max(10, round(opts.lrstep/2)): opts.epoch;
saveps = [saveps, 0: opts.lrstep: opts.epoch, opts.epoch];
[net, info] = train_imagenet(net, imdb, trainBatchFunc, testBatchFunc, ...
    'continue'       , opts.continue              , ...
    'debug'          , opts.debug                 , ...
    'plotStatistics' , opts.plot                  , ...
    'expDir'         , opts.expDir                , ...
    'batchSize'      , opts.batchSize             , ...
    'numEpochs'      , opts.epoch                 , ...
    'saveEpochs'     , unique(saveps)             , ...
    'learningRate'   , lrvec                      , ...
    'weightDecay'    , opts.wdecay                , ...
    'backPropDepth'  , opts.bpdepth               , ...
    'val'            , find(imdb.images.set == 3) , ...
    'gpus'           , opts.gpus                  , ...
    'errorFunction'  , 'none'                     , ...
    'epochCallback'  , @epoch_callback) ;

if ~isempty(opts.gpus)
    net = vl_simplenn_move(net, 'gpu'); 
end

% ---------------
% test
% ---------------
if 0 %~mod(opts.epoch, opts.testInterval)
    imdb_test = imdb;
    imdb_test.images = imdb.images.all;
    test_imagenet(net, imdb_test, testBatchFunc, opts, opts.metrics);
end
diary('off');
end


% -------------------------------------------------------------------
% postprocessing after each epoch
% -------------------------------------------------------------------
function net = epoch_callback(epoch, net, imdb, testBatchFunc)
opts = net.layers{end}.opts;
if numel(opts.gpus) >= 1
    net = vl_simplenn_move(net, 'gpu');
end
% disp
myLogInfo('[%s]', opts.methodID);
myLogInfo('[%s]', opts.identifier);
% test?
imdb_test = imdb;
imdb_test.images = imdb.images.all;
if ~mod(epoch, opts.testInterval) || epoch == opts.epoch
    % test full
    test_imagenet(net, imdb_test, testBatchFunc, opts, opts.metrics);
elseif epoch > 1 && ~mod(epoch, 20)
    % test random subset
    test_imagenet(net, imdb_test, testBatchFunc, opts, opts.metrics, ...
        false, [200 5e3]);
end
% slope annealing trick
if opts.sigmf(2) > 1
    opts.sigmf_p = min(opts.sigmf(1), opts.sigmf_p * opts.sigmf(2));
    net.layers{end}.opts = opts;
    myLogInfo('Sig = %g', opts.sigmf_p);
end
diary off, diary on
end
