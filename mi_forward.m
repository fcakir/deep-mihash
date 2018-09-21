function [top] = mi_forward(layer, bot, top)
% Implementation of Hashing with Mutual Information as in:
%
% "Hashing with Mutual Information", 
% Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
% (* equal contribution)
% arXiv:1803.00974 2018
%
% Please cite the paper if you use this code.
%
% This function implements the forward pass for mutual information objective.

Y = squeeze(layer.class); % Nx1
X = squeeze(bot.x);  % 1x1xBxN -> BxN, raw scores (logits) for each bit
[nbits, N] = size(X);

opts = layer.opts;
onGPU = numel(opts.gpus) > 0;

% -----------------------------------------------------------------------------
% 1. get NxN affinity matrix
% -----------------------------------------------------------------------------
if ~opts.unsupervised
    assert(size(Y, 1) == N); 
    Aff = affinity_binary(Y, Y, X, X, opts);
else
    assert(isvector(Y));
    assert(length(Y) == N);
    try
        Aff = opts.Aff(Y, Y) > 0;
    catch
        Aff = eye(N, 'logical');
    end
end
Xp  = logical(Aff - diag(diag(Aff)));
Xn  = ~Aff;
if onGPU
    Xp = gpuArray(Xp);
    Xn = gpuArray(Xn);
end

% -----------------------------------------------------------------------------
% 2. compute distances from hash codes
% -----------------------------------------------------------------------------
phi = 2*sigmf(X, [opts.sigmf_p 0]) - 1;  % RELAXED hash codes to interval [-1, 1]
hdist = (nbits - phi' * phi)/2;   

% -----------------------------------------------------------------------------
% 3. estimate discrete distributions
% -----------------------------------------------------------------------------
%Delta = nbits / opts.nbins;
%Cntrs = 0: Delta: nbits;
global Deltax
global Cntrsx

if isempty(Deltax)
	Deltax = nbits/opts.nbins;
end
if isempty(Cntrsx)
	Cntrsx = 0:Deltax:nbits;
end

Delta = Deltax;
Cntrs = Cntrsx;

L     = length(Cntrs);
prCp  = sum(Xp, 2) ./ (N-1);
prCn  = 1 - prCp;
pDCp  = zeros(N, L);
pDCn  = zeros(N, L);
if onGPU
    pDCp = gpuArray(pDCp);
    pDCn = gpuArray(pDCn);
end

% new version, better when L<N
cXp = sum(Xp, 2);
cXn = sum(Xn, 2);
nz_p = cXp > 0;
nz_n = cXn > 0;

for l = 1:L
    pulse = triPulse(hdist, Cntrs(l), Delta);  % NxN
    pDCp(:, l) = sum(pulse .* Xp, 2);
    pDCn(:, l) = sum(pulse .* Xn, 2);
	pDCp(nz_p,l) = pDCp(nz_p, l) ./ cXp(nz_p);
	pDCn(nz_n,l) = pDCn(nz_n, l) ./ cXn(nz_n);
end

% unnormalized distance distributions
upDCp = pDCp;
upDCn = pDCn;

% pD
pD = (pDCp + pDCn) ./ (N-1);

% normalize
sum_pDCp = sum(pDCp, 2);  nz_p = sum_pDCp > 0;
sum_pDCn = sum(pDCn, 2);  nz_n = sum_pDCn > 0;
pDCp(nz_p, :) = bsxfun(@rdivide, pDCp(nz_p, :), sum_pDCp(nz_p));
pDCn(nz_n, :) = bsxfun(@rdivide, pDCn(nz_n, :), sum_pDCn(nz_n));

% -----------------------------------------------------------------------------
% 4. compute entropies
% -----------------------------------------------------------------------------
y0 = zeros(size(pD));
if onGPU, y0 = gpuArray(y0); end
ent_D   = ent(pD, y0);  % H(D)
ent_D_C = prCp .* ent(pDCp, y0) + prCn .* ent(pDCn, y0);  % H(D|C)

% -----------------------------------------------------------------------------
% 5. loss
% -----------------------------------------------------------------------------
top.x = sum(single(ent_D - ent_D_C));  % maximize MI -> minimize -MI
top.aux = [];
top.aux.phi = phi;
top.aux.Xp = Xp;
top.aux.Xn = Xn;
top.aux.distance = hdist;
top.aux.prCp = prCp;
top.aux.prCn = prCn;
top.aux.pDCp = pDCp;
top.aux.pDCn = pDCn;
top.aux.pD   = pD;
top.aux.upDCp = upDCp;
top.aux.upDCn = upDCn;
top.aux.sum_upDCp = sum_pDCp;
top.aux.sum_upDCn = sum_pDCn;


end


function y = triPulse_old(D, mid, delta, onGPU)
% differently vectorized version
%
%   D: 1xN row vector of input data
% mid: Bx1 column vector of bin centers
%   y: BxN "pulse" matrix
assert(isvector(mid) & isvector(D));
if ~iscolumn(mid), mid = mid'; end
if ~isrow(D), D = D'; end

y = zeros(length(mid), length(D));
if onGPU, y = gpuArray(y); end

x_minus_mid = bsxfun(@minus, D, mid);
ind = bsxfun(@gt, D, mid-delta) & bsxfun(@le, D, mid+delta);
y(ind) = 1 - abs(x_minus_mid(ind))./delta;
end

% -----------------------------------------------------------------------------
% entropy
% -----------------------------------------------------------------------------
function H = ent(p, y0)
logp = y0;
logp(p>0) = log2(p(p>0));
H = -sum(p .* logp, 2);
end
