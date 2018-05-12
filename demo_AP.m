function paths = demo_AP(dataset, nbits, modelType, varargin)
% Implementation of Hashing with Mutual Information as in:
%
% "Hashing with Mutual Information", 
% Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
% (* equal contribution)
% arXiv:1803.00974 2018
%
% Please cite the paper if you use this code.
%
% This is the main function to run experiments.  
%
% INPUTS
%   dataset  - (string) in {'cifar', 'nuswide', 'labelme'}
%   nbits    - (int) length of binary code
%  	modelType- (string) in {'fc1', 'vggf', 'vggf_ft'} among others corresponding
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
opts = get_opts(dataset, nbits, modelType, varargin{:});
finishup = onCleanup(@cleanup);

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
% get model 
% -----------------------------------------------------------------------------
[net, opts] = get_model(opts);

% -----------------------------------------------------------------------------
% get dataset
% -----------------------------------------------------------------------------
global imdb
[imdb, opts, net] = get_imdb(imdb, opts, net);

% -----------------------------------------------------------------------------
% get batch sampling function
% -----------------------------------------------------------------------------
batchFunc = get_batchFunc(imdb, opts, net);

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
[net, info] = train_simplenn(net, imdb, batchFunc, ...
    'continue', opts.continue, ...
    'debug', opts.debug, ...
    'plotStatistics', opts.plot, ...
    'expDir', opts.expDir, ...
    'batchSize', opts.batchSize, ...
    'numEpochs', opts.epoch, ...
    'saveEpochs', unique(saveps), ...
    'learningRate', lrvec, ...
    'weightDecay', opts.wdecay, ...
    'backPropDepth', opts.bpdepth, ...
    'val', find(imdb.images.set == 3), ...
    'gpus', opts.gpus, ...
    'errorFunction', 'none', ...
    'epochCallback', @epoch_callback) ;

% -----------------------------------------------------------------------------
% return value
% -----------------------------------------------------------------------------
paths.diary 		= opts.diary_path;
paths.expfolder 	= opts.expDir;
end


% -----------------------------------------------------------------------------
% postprocessing after each epoch
% -----------------------------------------------------------------------------
function net = epoch_callback(epoch, net, imdb, batchFunc, netopts)
opts = net.layers{end}.opts;
if numel(opts.gpus) >= 1
    net = vl_simplenn_move(net, 'gpu');
end
% disp
myLogInfo('[%s]', opts.methodID);
myLogInfo('[%s]', opts.identifier);
% test?
if ~isfield(opts, 'testFunc'), opts.testFunc = @test_supervised; end
if ~isfield(opts, 'testInterval'), opts.testInterval = 10; end
if ~isfield(opts, 'metrics'), opts.metics = {'AP'}; end
if ~mod(epoch, opts.testInterval) ...
        || (isfield(opts, 'ep1') & opts.ep1 & epoch==1) || (epoch == opts.epoch)
    opts.testFunc(net, imdb, batchFunc, opts, opts.metrics);
end

diary off, diary on
end

% -----------------------------------------------------------------------------
% set learning rate
% -----------------------------------------------------------------------------
function lrvec = set_lr(opts)
if opts.lrdecay>0 && opts.lrdecay<1
    cur_lr = opts.lr;
    lrvec = [];
    while length(lrvec) < opts.epoch
        lrvec = [lrvec, ones(1, opts.lrstep)*cur_lr];
        cur_lr = cur_lr * opts.lrdecay;
    end
else
    lrvec = opts.lr;
end
end

% -----------------------------------------------------------------------------
% set model save checkpoints:
% the model files are saved at every epoch value as specified by 'saveps' below. 
% -----------------------------------------------------------------------------
function saveps = set_saveps(opts)
saveps = 0: max(10, round(opts.lrstep/2)): opts.epoch;
saveps = [saveps, 0: opts.lrstep: opts.epoch, opts.epoch];
end
