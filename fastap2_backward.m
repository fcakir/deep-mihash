function [bot] = ap_backward(layer, bot, top)

X = squeeze(bot.x);  % nbitsxN
[nbits, N] = size(X);

opts = layer.opts;
onGPU = numel(opts.gpus) > 0;

%nD     = top.aux.nD;
%ND     = top.aux.ND;
nDp    = top.aux.nDp;
nDn    = top.aux.nDn;
%NDp    = top.aux.NDp;
phi  = top.aux.phi;
Xp   = top.aux.Xp;
Xn   = top.aux.Xn;
hdist = top.aux.distance;

%Delta = top.aux.Delta;
%Cntrs = top.aux.Cntrs;

FDp = top.aux.FDp;
pDp = top.aux.pDp;
pDn = top.aux.pDn;
FD  = top.aux.FD;
prp = top.aux.prp;
prn = top.aux.prn; 
sum_nDp = top.aux.sum_nDp;
sum_nDn = top.aux.sum_nDn;

global Deltax
global Cntrsx
Delta = Deltax;
Cntrs = Cntrsx;

Np = sum(Xp, 2);
Nn = sum(Xn, 2);
L = length(Cntrs);

% 1. d(FastAP)/d(n+)
%{
NDn  = ND - NDp;
alph = nDp .* NDn ./ (ND.^2);
alph(isnan(alph)) = 0;
d_L_nDp = (NDp.*ND+nDp.*NDn)./(ND.^2) + alph * triu(ones(L), 1)'; 
%d_L_nDp = (NDp.*ND)./(ND.^2) + alph * triu(ones(L), 1)'; 
d_L_nDp = bsxfun(@rdivide, d_L_nDp, Np);
d_L_nDp(isnan(d_L_nDp)|isinf(d_L_nDp)) = 0;
%}
tem  = FDp .* pDp;
alph = (pDp .* FD - tem .* prp)./(FD.^2);
alph(isnan(alph)) = 0;
d_L_pDp = (pDp.*FD+FDp.*FD-tem.*prp)./(FD.^2) + alph * triu(ones(L), 1)';
d_L_pDp = d_L_pDp .* prp;
d_L_pDp(isnan(d_L_pDp)|isinf(d_L_pDp)) = 0;

% 2. d(FastAP)/d(n-)
%{
bet = -nDp .* NDp ./ (ND.^2);
bet(isnan(bet)) = 0;
d_L_nDn = bet * triu(ones(L))';
d_L_nDn = bsxfun(@rdivide, d_L_nDn, Np);
d_L_nDn(isnan(d_L_nDn)|isinf(d_L_nDn)) = 0;
%}

bet = -(FDp.*pDp)./(FD.^2);
bet(isnan(bet)) = 0;
bet = bet * triu(ones(L))';
d_L_pDn = (bet .* prn) .* prp;
d_L_pDn(isnan(d_L_pDn)|isinf(d_L_pDn)) = 0;

if true
for i = 1:N
	d_L_pDp(i, :) = (((repmat(-nDp(i,:),L,1)+eye(L)*sum_nDp(i))./(sum_nDp(i)^2)) ...
   						* d_L_pDp(i, :)')';
	d_L_pDn(i, :) = (((repmat(-nDn(i,:), L, 1)+eye(L)*sum_nDn(i))./(sum_nDn(i)^2)) ...
   						* d_L_pDn(i, :)')';
end
end

d_L_pDp(isnan(d_L_pDp)|isinf(d_L_pDp)) = 0;
d_L_pDn(isnan(d_L_pDn)|isinf(d_L_pDn)) = 0;

% 3. d(FastAP)/d(phi)
d_L_phi = zeros(nbits, N);
if onGPU, d_L_phi = gpuArray(d_L_phi); end

Np = sum(Xp, 2);
Nn = sum(Xn, 2);
invalid = (Np==0) | (Nn==0);

for l = 1:L
    % NxN matrix of delta_hat(i, j, l) for fixed l
    dpulse = triPulseDeriv(hdist, Cntrs(l), Delta(l));  % NxN
    ddp = dpulse .* Xp;  % delta_l^+(i, j)
    ddn = dpulse .* Xn;  % delta_l^-(i, j)

    % highly highly vectorized
	
	%alpha_p = d_L_pDp(:, l)./Np; alpha_p(invalid) = 0;
	%alpha_n = d_L_pDn(:, l)./Nn; alpha_n(invalid) = 0;
	%alpha_p = diag(alpha_p);
	%alpha_n = diag(alpha_n);
	
	
    alpha_p = diag(d_L_pDp(:, l));  
    alpha_n = diag(d_L_pDn(:, l));  
	%keyboard

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
%d_L_Delta = zeros(L, 1);
dlr = opts.dlr/N;
if true
	for l = 1:L
		dpulse = triPulseDerivDelta(hdist, Cntrs(l), Delta(l));
		ddp = sum(dpulse .* Xp, 2); % NX1
		ddn = sum(dpulse .* Xn, 2);
		d_L_Delta = gather((d_L_pDp(:, l))'*ddp + (d_L_pDn(:, l))'*ddn);
		d_L_Delta(isnan(d_L_Delta)|isinf(d_L_Delta)) = 0;
		Delta(l) = gather(max(opts.mind, Delta(l)+dlr*d_L_Delta));
		if randn > 0.99
			myLogInfo('%d: N:%.4f P:%.4f uP:%.4f', l, Deltax(l), Delta(l), d_L_Delta);
		end
	end	
	Deltax = Delta;
	% update
	%Deltax = gather(max(opts.mind, Deltax+dlr*sum(d_L_Delta)));
	%myLogInfo('%.2f+%.2f=%.2f', Deltax, dlr*sum(d_L_Delta), Deltax);
end

% Update centers
if true
	llr = opts.llr/N;
	for l = 1:L
		dpulse = -triPulseDeriv(hdist, Cntrs(l), Delta);
		ddp = sum(dpulse .* Xp, 2);
		ddn = sum(dpulse .* Xn, 2);
		d_L_l = (d_L_pDp(:, l)./Np)'*ddp + (d_L_pDn(:,l)./Nn)'*ddn;
		d_L_l(isnan(d_L_l)|isinf(d_L_l)) = 0;
		% update
		Cntrs(l) = gather(min(max(0, Cntrs(l) + llr*d_L_l), nbits));

		if true & randn > 0.99
			myLogInfo('%d: N:%.4f P:%.4f uP:%.4f', l, Cntrsx(l), Cntrs(l), d_L_l);
			if false & abs(d_L_l) > 10
				keyboard
			end
		end
	end
	Cntrsx = Cntrs;
end

end
