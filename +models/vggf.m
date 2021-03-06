function [net, imageSize, normalize] = vggf(opts)
% -----------------------------------------------------------------------------
% Initializes a pre-trained VGG-F net for transfer learning. 
% -----------------------------------------------------------------------------
imageSize = 224;
normalize = false;
lr = [1 1];

% -----------------------------------------------------------------------------
% finetune VGG-F
% -----------------------------------------------------------------------------
net = load(fullfile(opts.localDir, 'models', 'imagenet-vgg-f.mat'));
net.layers(end) = [];
net.layers(end) = [];

% -----------------------------------------------------------------------------
% insert dropout layers (removed in deploy model)
% -----------------------------------------------------------------------------
if opts.dropout > 0
    assert(strcmp(net.layers{end}.name, 'relu7'));
    net.layers(end:end+1) = net.layers(end-1:end);
    net.layers{end-2} = struct('type', 'dropout', 'name', 'drop6', ...
        'rate', opts.dropout);
    net.layers{end+1} = struct('type', 'dropout', 'name', 'drop7', ...
        'rate', opts.dropout);
end

% -----------------------------------------------------------------------------
% fully connected layer
% -----------------------------------------------------------------------------
net.layers{end+1} = struct('type', 'conv', 'name', 'logits', ...
    'weights', {models.init_weights(1, 4096, opts.nbits)}, ...
    'learningRate', lr, ...
    'stride', 1, 'pad', 0);

% -----------------------------------------------------------------------------
% loss layer
% -----------------------------------------------------------------------------
net.layers{end+1} = struct('type', 'custom', 'name', 'loss', ...
    'opts', opts, 'weights', [], 'precious', false, ...
    'forward', str2func([opts.obj '_forward']), ...
    'backward', str2func([opts.obj, '_backward']));

cellfun(@(x) x.name, net.layers, 'uniform', 0)
% -----------------------------------------------------------------------------
% Fill in default values
% -----------------------------------------------------------------------------
net = vl_simplenn_tidy(net);
end
