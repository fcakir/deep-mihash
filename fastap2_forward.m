function [top] = ap_forward(layer, bot, top)
Y = squeeze(layer.class); % Nx1
X = squeeze(bot.x);  % 1x1xBxN -> BxN, raw scores (logits) for each bit
[nbits, N] = size(X);

opts = layer.opts;
onGPU = numel(opts.gpus) > 0;

% get NxN affinity matrix
if ~opts.unsupervised, 
    assert(size(Y, 1) == N); 
    Aff = affinity_binary(Y, Y, X, X, opts);
else
    assert(isvector(Y));
    assert(length(Y) == N);
    Aff = opts.Aff(Y, Y);
end
Xp = logical(Aff - diag(diag(Aff)));
Xn = ~Aff;
if onGPU
    Xp = gpuArray(Xp);
    Xn = gpuArray(Xn);
end

% compute distances from hash codes
phi = 2*sigmf(X, [opts.sigmf_p 0]) - 1;  % RELAXED hash codes to interval [-1, 1]
hdist = (nbits - phi' * phi)/2;   

% estimate discrete distributions
%{
if 1
    Delta = opts.nbits / opts.nbins;
    Cntrs = 0: Delta: opts.nbits;
else
    dmin  = min(hdist(:)); 
    dmax  = max(hdist(:));
    Delta = (dmax-dmin) / opts.nbins;
    Cntrs = dmin: Delta: dmax;
end
%}
global Deltax
global Cntrsx

if isempty(Deltax)
	Deltax = nbits/opts.nbins;
end
if isempty(Cntrsx)
	Cntrsx = 0:Deltax:nbits;
end

if length(Deltax) == 1
	Deltax = repmat(Deltax, 1, length(Cntrsx));
end
Delta = Deltax;
Cntrs = Cntrsx;

L   = length(Cntrs); 
prp  = sum(Xp, 2) ./ (N-1); % P(+) Nx1
prn  = 1 - prp; % P(-) Nx1
nD  = zeros(N, L);
nDp = zeros(N, L);
nDn = zeros(N, L);
nDp2 = zeros(N, L);
nDn2 = zeros(N, L);
pD  = zeros(N, L);
pDp = zeros(N, L);
pDn = zeros(N, L);

if onGPU
    nD  = gpuArray(nD);
    nDp = gpuArray(nDp);
    nDn = gpuArray(nDn);
	nDp2 = gpuArray(nDp2);
	nDn2 = gpuArray(nDn2);
    pD  = gpuArray(pD);
    pDp = gpuArray(pDp);
    pDn = gpuArray(pDn);
end


Np = sum(Xp, 2);
Nn = sum(Xn, 2);
nz_p = Np > 0;
nz_n = Nn > 0;

% new version, better when L<N
for l = 1:L
    pulse = triPulse(hdist, Cntrs(l), Delta(l));  % NxN
    nDp(:, l) = sum(pulse .* Xp, 2);
    nDn(:, l) = sum(pulse .* Xn, 2);
	%nDp(nz_p, l) = nDp(nz_p, l) ./ Np(nz_p);
	%nDn(nz_n, l) = nDn(nz_n, l) ./ Nn(nz_n);
end

%pD = (nDp + nDn) ./ (N-1);

sum_nDp = sum(nDp, 2);  nz_p = sum_nDp > 0;
sum_nDn = sum(nDn, 2);  nz_n = sum_nDn > 0;
pDp(nz_p, :) = bsxfun(@rdivide, nDp(nz_p, :), sum_nDp(nz_p));
pDn(nz_n, :) = bsxfun(@rdivide, nDn(nz_n, :), sum_nDn(nz_n));
%pDp = nDp;
%pDn = nDn;
pD = pDp .* prp + pDn .* prn;
%nD  = nDp + nDn;
%ND  = cumsum(nD, 2);
%NDp = cumsum(nDp, 2);

FD = cumsum(pD, 2);
FDp = cumsum(pDp, 2); 
%keyboard
% compute FastAP
%fastap = nDp .* NDp ./ ND;
fastap = pDp .* FDp ./ FD; 
fastap(isnan(fastap)) = 0;
fastap = sum(fastap, 2) .* prp;
fastap(isnan(fastap)) = 0;

% loss
top.x = sum(fastap);
top.aux = [];
top.aux.Cntrs = Cntrs;
top.aux.Delta = Delta;
top.aux.distance = hdist;
top.aux.phi = phi;
top.aux.Xp = Xp;
top.aux.Xn = Xn;
%top.aux.nD = nD;
%top.aux.ND = ND;
top.aux.nDp = nDp;
top.aux.nDn = nDn;
%top.aux.NDp = NDp;

top.aux.pDp = pDp;
top.aux.pDn = pDn;
top.aux.FD  = FD;
top.aux.FDp = FDp;
top.aux.prp = prp;
top.aux.prn = prn;
top.aux.sum_nDp = sum_nDp;
top.aux.sum_nDn = sum_nDn;
end
