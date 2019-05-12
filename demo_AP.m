function paths = demo_AP(dataset, nbits, modelType, varargin)
% This is the main function to run deep learning experiments for the mutual 
% information based hashing method as described in the below papers. 
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
% refer to github page
% 
% -----------------------------------------------------------------------------
% initialize opts
% -----------------------------------------------------------------------------
opts = get_opts(dataset, nbits, modelType, varargin{:});
finishup = onCleanup(@cleanup);
rng(opts.randseed, 'twister'); % set global random stream

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
[imdb, opts, net] = get_imdb(imdb, opts, net);

% -----------------------------------------------------------------------------
% HBMP codes
% -----------------------------------------------------------------------------
if strcmpi(opts.obj, 'hbmp')
	if ~isfield(imdb.images,'greedy_labels')
		if ~isfield(imdb.images, 'orig_labels')
			myLogInfo('Doing binary inference, learning from scratch');
			opts.continue = false;
			imdb = hbmp_codes(imdb, opts);
		end
	else
		imdb.images.labels = imdb.images.greedy_labels;
	end
end
% -----------------------------------------------------------------------------
% set batch sampling function
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
    'plotStatistics', false, ...
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
% swap the hbmp codes with original labels
% -----------------------------------------------------------------------------
if strcmpi(opts.obj, 'hbmp')
    imdb.images.greedy_labels = imdb.images.labels;
    imdb.images.labels = imdb.images.orig_labels;
end
% -----------------------------------------------------------------------------
% return 
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
if ~isfield(opts, 'metrics'), opts.metrics = {'AP'}; end
if ~mod(epoch, opts.testInterval) ...
        || (isfield(opts, 'ep1') & opts.ep1 & epoch==1) || (epoch == opts.epoch)
    opts.testFunc(net, imdb, batchFunc, opts, opts.metrics);
end

diary off, diary on
end


