function [bot] = ap_backward(layer, bot, top)

X = squeeze(bot.x);  % nbitsxN
[nbits, N] = size(X);

opts = layer.opts;
onGPU = numel(opts.gpus) > 0;

nD     = top.aux.nD;
ND     = top.aux.ND;
nDp    = top.aux.nDp;
nDn    = top.aux.nDn;
NDp    = top.aux.NDp;
phi  = top.aux.phi;
Xp   = top.aux.Xp;
Xn   = top.aux.Xn;
hdist = top.aux.distance;
Delta = top.aux.Delta;
Cntrs = top.aux.Cntrs;


%global Deltax
%global Cntrsx
%Delta = Deltax;
%Cntrs = Cntrsx;

Np = sum(Xp, 2);
Nn = sum(Xn, 2);
L = length(Cntrs);

% 1. d(FastAP)/d(n+)
NDn  = ND - NDp;
alph = nDp .* NDn ./ (ND.^2);
alph(isnan(alph)) = 0;
d_L_nDp = (NDp.*ND+nDp.*NDn)./(ND.^2) + alph * triu(ones(L), 1)'; 
%d_L_nDp = (NDp.*ND)./(ND.^2) + alph * triu(ones(L), 1)'; 
d_L_nDp = bsxfun(@rdivide, d_L_nDp, Np);
d_L_nDp(isnan(d_L_nDp)|isinf(d_L_nDp)) = 0;

% 2. d(FastAP)/d(n-)
bet = -nDp .* NDp ./ (ND.^2);
bet(isnan(bet)) = 0;
d_L_nDn = bet * triu(ones(L))';
d_L_nDn = bsxfun(@rdivide, d_L_nDn, Np);
d_L_nDn(isnan(d_L_nDn)|isinf(d_L_nDn)) = 0;

% 3. d(FastAP)/d(phi)
d_L_phi = zeros(nbits, N);
if onGPU, d_L_phi = gpuArray(d_L_phi); end
for l = 1:L
    % NxN matrix of delta_hat(i, j, l) for fixed l
    dpulse = triPulseDeriv(hdist, Cntrs(l), Delta);  % NxN
    ddp = dpulse .* Xp;  % delta_l^+(i, j)
    ddn = dpulse .* Xn;  % delta_l^-(i, j)

    % highly highly vectorized
    alpha_p = diag(d_L_nDp(:, l));  
    alpha_n = diag(d_L_nDn(:, l));  
    Ap = ddp * alpha_p + alpha_p * ddp;  % 1st term: j=i, 2nd term: j~=i
    An = ddn * alpha_n + alpha_n * ddn;

    % accumulate gradient
    d_L_phi = d_L_phi - 0.5 * phi * (Ap + An);
end

% 4. d(FastAP)/d(x)
sigmoid = (phi+1)/2;
d_phi_x = 2 .* sigmoid .* (1-sigmoid) * opts.sigmf_p;  % nbitsxN
d_L_x   = -d_L_phi .* d_phi_x;

% 5. final
bot.dzdx = zeros(size(bot.x), 'single');
if onGPU, bot.dzdx = gpuArray(bot.dzdx); end
bot.dzdx(1, 1, :, :) = single(d_L_x);


% Update Delta
d_L_Delta = 0;
dlr = opts.dlr/N; %TODO: should this be here, yes if you're doing SGD internally.
if false
	for l = 1:L
		dpulse = triPulseDerivDelta(hdist, Cntrs(l), Delta);
		ddp = sum(dpulse .* Xp, 2); % NX1
		ddn = sum(dpulse .* Xn, 2);
		d_L_Delta = d_L_Delta + d_L_nDp(:, l)'*ddp + d_L_nDn(:, l)'*ddn;
	end	
	Deltax = max(opts.mind, Deltax + dlr*d_L_Delta);
	myLogInfo('%.2f-%.2f', Deltax+dlr*d_L_Delta, Deltax);
end

% Update centers
if false
	llr = opts.llr/N;
	for l = 1:L
		dpulse = -triPulseDeriv(hdist, Cntrs(l), Delta);
		ddp = dpulse .* Xp;
		ddn = dpulse .* Xn;
		d_L_l = d_L_nDp(:, l)'*ddp + d_L_nDn(:,l)'*ddn;
		Cntrs(l) = min(max(eps, Cntrs(l) - llr*d_L_l), nbits-eps);
	end
	if randn > 0.99
		l = randi([1, L], 1);
		myLogInfo('%d: %.2f-%.2f',l, Cntrsx(l), Cntrs(l));
	end
	Cntrsx = Cntrs;
end

end
