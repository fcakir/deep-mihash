function [net, imageSize, normalize] = fc1(opts)
% -----------------------------------------------------------------------------
% Initializes a one fully connected layer. 
% -----------------------------------------------------------------------------
imageSize = 0;
if isfield(opts,  'normalize')
    normalize = opts.normalize;
else
    normalize = true;
end

if normalize
    lr = [1 0.1] ;
else
    lr = [1 1];
end

% -----------------------------------------------------------------------------
% LabelMe by default uses 512 dimensional GIST features, other datasets uses 
% the penultimate layer of a VGG architecture which has 4096 dimensions. 
% -----------------------------------------------------------------------------
if strcmp(opts.dataset, 'labelme')
    D = 512;
else
    D = 4096;
end
net.layers = {} ;

% -----------------------------------------------------------------------------
% fully connected layer (logits for each bit)
% -----------------------------------------------------------------------------
net.layers{end+1} = struct('type', 'conv', ...
    'name', 'fc1', ...
    'weights', {models.init_weights(1,D,opts.nbits)}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 0) ;

% -----------------------------------------------------------------------------
% loss layer
% -----------------------------------------------------------------------------
net.layers{end+1} = struct('type', 'custom', ...
    'name', 'loss', ...
    'opts', opts, ...
    'forward', str2func([opts.obj '_forward']), ...
    'backward', str2func([opts.obj, '_backward']));
net.layers{end}.precious = false;
net.layers{end}.weights = {};

% -----------------------------------------------------------------------------
% Meta parameters
% -----------------------------------------------------------------------------
net.meta.inputSize = [1 1 D] ;

% -----------------------------------------------------------------------------
% Fill in default values
% -----------------------------------------------------------------------------
net = vl_simplenn_tidy(net) ;
end
