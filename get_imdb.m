function [imdb, opts, net] = get_imdb(imdb, opts, net)

% -----------------------------------------------------------------------------
% feature type:
% For the fully connected layer model (fc) the input is a feature descriptor 
% such as GIST. 
% For LabelMe the GIST descriptor is used. The fc7 layer features are used for
% other datasets.
% -----------------------------------------------------------------------------
if ~isempty(strfind(opts.modelType, 'fc')) || opts.imageSize <= 0
    if strcmp(opts.dataset, 'labelme')
        imdbName = sprintf('%s_gist', opts.dataset);
    else
        imdbName = sprintf('%s_fc7', opts.dataset);
    end
else
    imdbName = opts.dataset;
end
imdbFunc = str2func(['imdb.' imdbName]);

% -----------------------------------------------------------------------------
% normalize images/features?
% If features are used (opposed to images) as input, then you can normalize them
% -----------------------------------------------------------------------------
if ismember(opts.imageSize, [224 227])
    assert(~opts.normalize);
end
if opts.normalize
    imdbName = [imdbName '_normalized'];
end
if ismember(opts.dataset, {'cifar', 'nus'})
    imdbName = sprintf('%s_split%d', imdbName, opts.split);
end
if strcmp(opts.dataset, 'imagenet')
    imdbName = sprintf('%s_%dx%d', imdbName, opts.imageSize, opts.imageSize);
end
% -----------------------------------------------------------------------------
% imdbName finalized
% -----------------------------------------------------------------------------
if ~isempty(imdb) && strcmp(imdb.name, imdbName)
    myLogInfo('%s already loaded', imdb.name);
    return;
end
imdb = [];

% -----------------------------------------------------------------------------
% imdbFile
% checks the {opts.dataDir}/IMDB-FILES/ location for imdb mat file. 
% -----------------------------------------------------------------------------
imdbFile = fullfile(opts.dataDir, 'IMDB-FILES', ['imdb_' imdbName]);
imdbFile = [imdbFile, '.mat'];
myLogInfo(imdbFile);

% -----------------------------------------------------------------------------
% load/save
% -----------------------------------------------------------------------------
t0 = tic;
if exist(imdbFile, 'file')
    imdb = load(imdbFile) ;
    myLogInfo('loaded in %.2fs', toc(t0));
else
    imdb = imdbFunc(opts, net) ;
    save(imdbFile, '-struct', 'imdb', '-v7.3') ;
    unix(['chmod g+rw ' imdbFile]); 
    myLogInfo('saved in %.2fs', toc(t0));
end
imdb.name = imdbName;
imdb.filepath = imdbFile;
myLogInfo('%s loaded', imdb.name);

% -----------------------------------------------------------------------------
% is dataset unsupervised ?
% If the dataset is unsupervised, the affinity matrix of the training data is
% pre-computed to be used in the last layer. See mi_forward.m .  
% -----------------------------------------------------------------------------
if isempty(imdb.images.labels)		
    assert(opts.unsupervised & isfield(imdb.images, 'thr_dist'));
    opts.thr_dist = imdb.images.thr_dist;
    itrain = find(imdb.images.set == 1);
    Xtrain = squeeze(imdb.images.data(:, :, :, itrain))';
    opts.Aff = affinity_binary([], [], Xtrain, Xtrain, opts);
    net.layers{end}.opts = opts;
end

end
