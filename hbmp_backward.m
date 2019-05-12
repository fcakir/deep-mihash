function [bot] = hbmp_backward(layer, bot, top)
Y = top.aux.Y';
X = squeeze(bot.x);  % nbitsxN
[nbits, N] = size(X);

opts = layer.opts;
onGPU = numel(opts.gpus) > 0;

bot.dzdx = zeros(size(bot.x), 'single');
if onGPU, bot.dzdx = gpuArray(bot.dzdx); end
Z = X.*Y;
Y(Z>1) = 0;
bot.dzdx(1,1,:,:) = -single(Y);
end
