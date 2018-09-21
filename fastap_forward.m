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

if 1
    Delta = opts.nbits / opts.nbins;
    Cntrs = 0: Delta: opts.nbits;
else
    dmin  = min(hdist(:)); 
    dmax  = max(hdist(:));
    Delta = (dmax-dmin) / opts.nbins;
    Cntrs = dmin: Delta: dmax;
end
%{
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
%}
L   = length(Cntrs); 
nD  = zeros(N, L);
nDp = zeros(N, L);
nDn = zeros(N, L);
if onGPU
    nD  = gpuArray(nD);
    nDp = gpuArray(nDp);
    nDn = gpuArray(nDn);
end

% new version, better when L<N
for l = 1:L
    pulse = triPulse(hdist, Cntrs(l), Delta);  % NxN
    nDp(:, l) = sum(pulse .* Xp, 2);
    nDn(:, l) = sum(pulse .* Xn, 2);
end
nD  = nDp + nDn;
ND  = cumsum(nD, 2);
NDp = cumsum(nDp, 2);

% compute FastAP
fastap = nDp .* NDp ./ ND;
fastap(isnan(fastap)) = 0;
fastap = sum(fastap, 2) ./ sum(Xp, 2);
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
top.aux.nD = nD;
top.aux.ND = ND;
top.aux.nDp = nDp;
top.aux.nDn = nDn;
top.aux.NDp = NDp;
end
