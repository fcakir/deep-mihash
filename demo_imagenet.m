function paths = demo_imagenet(nbits, modelType, varargin)
% This is the main function to run deep learning experiments for the mutual 
% information based hashing method as described in the below papers on ImageNet100. 
%
% Please cite these papers if you use this code.
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
%   nbits    - (int) length of binary code
%  	modelType- (string) in {'alexnet_ft', 'vggf_ft'} among others corresponding
%			   to the models as defined under '+models' folder. 
%   varargin - key-value argument pairs, see get_opts.m for details
%
% OUTPUTS
%   paths (struct)
%       .expfolder - (string) Path to the experiments folder
%       .diary     - (string) Path to the experimental log
%
% EXAMPLE COMMANDS
% 	

% -----------------------------------------------------------------------------
% initialize opts
% -----------------------------------------------------------------------------
opts = get_opts('imagenet', nbits, modelType, 'split', 1, varargin{:});
cleanupObj = onCleanup(@cleanup);
rng(opts.randseed, 'twister'); % set global random stream

% hard-coded fields
assert(ismember(opts.modelType, {'alexnet_ft', 'vggf_ft'}));
opts.metrics = {'AP@1000'};
opts.testFunc = @test_imagenet;

% -----------------------------------------------------------------------------
% post-parsing
% -----------------------------------------------------------------------------
opts = process_opts(opts);  % carry out all post-processing on opts

% -----------------------------------------------------------------------------
% print info
% -----------------------------------------------------------------------------
opts
myLogInfo(opts.methodID);
myLogInfo(opts.identifier);

% -----------------------------------------------------------------------------
% get neural net model 
% -----------------------------------------------------------------------------
[net, opts] = get_model(opts);

% -----------------------------------------------------------------------------
% get dataset
% -----------------------------------------------------------------------------
global imdb
imdb = get_imdb(imdb, opts, net);

% -----------------------------------------------------------------------------
% set batch sampling function
% -----------------------------------------------------------------------------
testBatchFunc   = get_batchFunc(imdb, opts, net);
trainBatchFunc  = @batch_simplenn;

% -----------------------------------------------------------------------------
% set learning rate vector
% -----------------------------------------------------------------------------
lrvec = set_lr(opts);

% -----------------------------------------------------------------------------
% set model save checkpoints
% -----------------------------------------------------------------------------
saveps = set_saveps(opts);

% -----------------------------------------------------------------------------
% train
% -----------------------------------------------------------------------------
[net, info] = train_imagenet(net, imdb, trainBatchFunc, testBatchFunc, ...
    'continue'       , opts.continue              , ...
    'debug'          , opts.debug                 , ...
    'plotStatistics' , false                      , ...
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

% -----------------------------------------------------------------------------
% return 
% -----------------------------------------------------------------------------
paths.diary 		= opts.diary_path;
paths.expfolder 	= opts.expDir;
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
if ~mod(epoch, opts.testInterval) ...
   || (isfield(opts, 'ep1') & opts.ep1 & epoch==1) || (epoch == opts.epoch)
    % test full
    test_imagenet(net, imdb_test, testBatchFunc, opts, opts.metrics);
end

diary off, diary on
end

