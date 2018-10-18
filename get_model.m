function [net, opts] = get_model(opts)
% -----------------------------------------------------------------------------
% calls the appropiate function in +models for neural network model construction
% -----------------------------------------------------------------------------
t0 = tic;
modelFunc = str2func(sprintf('models.%s', opts.modelType));
[net, opts.imageSize, opts.normalize] = modelFunc(opts);
myLogInfo('%s in %.2fs', opts.modelType, toc(t0));
end
