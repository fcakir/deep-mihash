function opts = process_opts(opts)
% post-parse processing

% -----------------------------------------------------------------------------
% dataset
% -----------------------------------------------------------------------------
if strcmp(opts.dataset, 'cifar')
    opts.testFunc = @test_supervised;
    opts.unsupervised = false;
	assert(ismember(opts.modelType, {'vggf', 'fc1'}));
elseif strcmp(opts.dataset, 'nus')
    opts.testFunc = @test_supervised;
    opts.unsupervised = false;
    if strcmp(opts.modelType, 'vggf')
        myLogInfo('Overriding modelType vggf with vggf_ft');
        opts.modelType = 'vggf_ft';
    end
	assert(ismember(opts.modelType, {'vggf_ft', 'fc1'}));
elseif strcmp(opts.dataset, 'labelme')
    opts.testFunc = @test_unsupervised;
    opts.unsupervised = true;
	assert(ismember(opts.modelType == 'fc1'));
elseif strcmp(opts.dataset, 'imagenet') % for future purposes 
    opts.testFunc = @test_supervised;
    opts.unsupervised = false;
	assert(opts.modelType == 'alexnet_ft');
else, error('unknown dataset');
end

% -----------------------------------------------------------------------------
% modelType
% -----------------------------------------------------------------------------
if strcmp(opts.modelType, 'fc1')
    opts.bpdepth = inf;
    opts.dropout = 0;
end
if ismember(opts.modelType, {'alexnet_ft', 'vggf', 'vggf_ft'})
    opts.normalize = false;
    assert(numel(opts.gpus) > 0, 'set ''gpus'' for non-single layer architectures');
end

% -----------------------------------------------------------------------------
% assertions
% -----------------------------------------------------------------------------
assert(opts.bpdepth >= 2);
assert(opts.dropout >= 0 && opts.dropout < 1);
assert(~isempty(opts.metrics));

% -----------------------------------------------------------------------------
% sigmoid approx
% -----------------------------------------------------------------------------
assert(opts.sigmf(1) >= 1);
assert(opts.sigmf(2) == 0);
opts.sigmf_p = opts.sigmf(1);

% -----------------------------------------------------------------------------
% identifier
% -----------------------------------------------------------------------------
% methodID
opts.methodID = sprintf('%s-%s%d-%s', upper(opts.obj), opts.dataset, ...
    opts.nbits, opts.modelType);
if ismember(opts.dataset, {'cifar' 'nus'})
    opts.methodID = sprintf('%s-sp%d', opts.methodID, opts.split);
end

% 0. shared
idr = sprintf('Bin%dSig%g,%g-batch%d-%sLR%gD%g', ...
	opts.nbins, opts.sigmf(1), opts.sigmf(2), opts.batchSize, opts.solver, ...
	opts.lr, opts.lrdecay);
% 1. lrstep
if opts.lrdecay > 0
    idr = sprintf('%sE%d', idr, opts.lrstep);
end
% 2. weight decay
if opts.wdecay > 0
    idr = sprintf('%s-wdecay%g', idr, opts.wdecay);
end
% 3. backprop depth
if opts.bpdepth < inf
    idr = sprintf('%s-BP%d', idr, opts.bpdepth);
end
% 4. dropout
if opts.dropout > 0 && opts.dropout < 1
    idr = sprintf('%s-Dropout%g', idr, opts.bpdepth);
end
% 5. VGG-F on NUS and AlexNet on ImageNet100
%    lr multiplier for pretrained layers
if ismember(opts.modelType, {'vggf_ft' 'alexnet_ft'})
    if opts.lrmult > 1 || opts.lrmult < 0
        opts.lrmult = max(0, min(opts.lrmult, 1));
        myLogInfo('Warning: clipping opts.lrmult to [0, 1]');
        myLogInfo('         opts.lrmult = %g', opts.lrmult);
    end
    idr = sprintf('%s-lrmult%g', idr, opts.lrmult);
end
% 6. feature normalized?
if opts.normalize
    idr = [idr, '-ftnorm']; 
end

opts.identifier = idr;

% -----------------------------------------------------------------------------
% generic
% -----------------------------------------------------------------------------
opts.localDir = './cachedir'; 
if ~exist(opts.localDir, 'file')
	error('Please make a symlink/folder for cachedir!');
end
opts.dataDir = fullfile(opts.localDir, 'data');
opts.imdbPath = fullfile(opts.dataDir, [opts.dataset '_imdb']);

% -----------------------------------------------------------------------------
% expDir: string such as ./cachedir/results/MI-cifar32-vggf-sp1
% -----------------------------------------------------------------------------
opts.expDir = fullfile(opts.localDir, 'results', opts.methodID);
if exist(opts.expDir, 'dir') == 0, 
    mkdir(opts.expDir);
    unix(['chmod g+rw ' opts.expDir]); 
end

% -----------------------------------------------------------------------------
% identifier string for the current experiment
% NOTE: opts.identifier is already initialized with method-specific params
% -----------------------------------------------------------------------------
idr = opts.identifier;
if isempty(opts.prefix)
    % prefix: timestamp
	% Folder must be a GIT repo with a commit history
    [~, T] = unix(['git log -1 --format=%ci|cut -d " " -f1,2|cut -d "-" -f2,3' ...
        '|tr " " "."|tr -d ":-"']);
    opts.prefix = strrep(T, char(10), '');
end
opts.identifier = [opts.prefix '-' idr];  % remove \n

% -----------------------------------------------------------------------------
% expand expDir to:
% ./cachedir/results/MI-cifar32-vggf-sp1/abcdef-Bin16Sig30,0-batch256-......
% -----------------------------------------------------------------------------
opts.expDir = fullfile(opts.expDir, opts.identifier);
if ~exist(opts.expDir, 'dir'),
    myLogInfo(['creating opts.expDir: ' opts.expDir]);
    mkdir(opts.expDir);
    unix(['chmod g+rw ' opts.expDir]); 
end

% -----------------------------------------------------------------------------
% set up GPU
% -----------------------------------------------------------------------------
if ~isempty(opts.gpus) && opts.gpus == 0
	myLogInfo('Setting up GPU');
    opts.gpus = auto_select_gpu;
end

% -----------------------------------------------------------------------------
% record a diary
% -----------------------------------------------------------------------------
opts.diary_path = record_diary(opts);
end
