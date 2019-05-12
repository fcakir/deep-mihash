function [top] = hbmp_forward(layer, bot, top)
Y = squeeze(layer.class); % N x nbits
%assert(~any(Y(:) == -2));
X = squeeze(bot.x);  % 1x1xBxN -> BxN, raw scores (logits) for each bit
[nbits, N] = size(X);

opts = layer.opts;
if ~opts.unsupervised, assert(size(Y, 1) == N); end;
onGPU = numel(opts.gpus) > 0;

% loss
Z = hinge(X.*Y');
top.x = sum(Z(:));
top.aux.Y = Y;
end

function out = hinge(z)
out = max(0, 1-z);
end
