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
Delta = nbits / opts.nbins;
Cntrs = 0: Delta: nbits;
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
for l = 1:L
    pulse = triPulse(hdist, Cntrs(l), Delta);  % NxN
    pDCp(:, l) = sum(pulse .* Xp, 2);
    pDCn(:, l) = sum(pulse .* Xn, 2);
end

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
end

% -----------------------------------------------------------------------------
% entropy
% -----------------------------------------------------------------------------
function H = ent(p, y0)
logp = y0;
logp(p>0) = log2(p(p>0));
H = -sum(p .* logp, 2);
end

% -----------------------------------------------------------------------------
% triangular kernel function used for differentiable histogram binning
% -----------------------------------------------------------------------------
function y = triPulse(D, mid, delta)
% -----------------------------------------------------------------------------
%     D: input matrix of distance values
%   mid: scalar, the center of some histogram bin
% delta: scalar, histogram bin width
%
% For histogram bin mid, compute the contribution y ("pulse") 
% from every element in D.  
% Interpolation using the triangular kernel
% -----------------------------------------------------------------------------
ind = (mid-delta < D) & (D <= mid+delta);
y   = 1 - abs(D - mid) / delta;
y   = y .* ind;
end
