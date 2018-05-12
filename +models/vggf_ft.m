function [net, imageSize, normalize] = vggf_nus(opts)
% Initializes a pre-trained VGG-F net for transfer learning, for NUSWIDE.  
imageSize = 224;
normalize = false;
lr = [1 0.1];

% finetune VGG-F
net = load(fullfile(opts.localDir, 'models', 'imagenet-vgg-f.mat'));
net.layers(end) = [];
net.layers(end) = [];

n = numel(net.layers);
for i=1:n
	if (strcmp(net.layers{i}.type, 'conv'))
		net.layers{i}.dilate = 1;
	end
end

if opts.dropout > 0
    % insert dropout layers (removed in deploy model)
    assert(strcmp(net.layers{end}.name, 'relu7'));
    net.layers(end:end+1) = net.layers(end-1:end);
    net.layers{end-2} = struct('type', 'dropout', 'name', 'drop6', ...
        'rate', opts.dropout);
    net.layers{end+1} = struct('type', 'dropout', 'name', 'drop7', ...
        'rate', opts.dropout);
end
% test
if opts.lrmult < 1 && opts.lrmult >= 0
    for i = 1:length(net.layers)
        net.layers{i}.learningRate = opts.lrmult * [1 1];
    end
end

% FC layer
net.layers{end+1} = struct('type', 'conv', 'name', 'logits', ...
    'weights', {models.init_weights(1, 4096, opts.nbits)}, ...
    'learningRate', lr, ...
    'stride', 1, 'pad', 0);

% loss layer
net.layers{end+1} = struct('type', 'custom', 'name', 'loss', ...
    'opts', opts, 'weights', [], 'precious', false, ...
    'forward', str2func([opts.obj '_forward']), ...
    'backward', str2func([opts.obj, '_backward']));

cellfun(@(x) x.name, net.layers, 'uniform', 0)
net = vl_simplenn_tidy(net);
end
