function [top] = hbmp_forward(layer, bot, top)
%
% Please cite these papers if you use this code.
%
% 1. "Hashing with Binary Matrix Pursuit", 
%    Fatih Cakir, Kun He, Stan Sclaroff
%    European Conference on Computer Vision (ECCV) 2018
%    arXiV:1808.01990 
%
% 2. "Hashing with Mutual Information", 
%    Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
% 	 IEEE TPAMI 2019 (to appear)
%    arXiv:1803.00974
%
% 3. "MIHash: Online Hashing with Mutual Information", 
%    Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
%    International Conference on Computer Vision (ICCV) 2017
%    (* equal contribution)
%
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
