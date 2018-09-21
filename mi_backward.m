function [bot] = mi_backward(layer, bot, top)
% Implementation of Hashing with Mutual Information as in:
%
% "Hashing with Mutual Information", 
% Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
% (* equal contribution)
% arXiv:1803.00974 2018
%
% Please cite the paper if you use this code.
%
% This function implements the backward pass for mutual information objective.

X = squeeze(bot.x);  % nbitsxN
[nbits, N] = size(X);

opts = layer.opts;
onGPU = numel(opts.gpus) > 0;

global Deltax
global Cntrsx
Delta = Deltax;
Cntrs = Cntrsx;

%Delta = nbits / opts.nbins;
%Cntrs = 0: Delta: nbits;
L = length(Cntrs);

pD   = top.aux.pD;
pDCp = top.aux.pDCp;
pDCn = top.aux.pDCn;
prCp = top.aux.prCp;
prCn = top.aux.prCn;
phi  = top.aux.phi;
Xp   = top.aux.Xp;
Xn   = top.aux.Xn;
hdist = top.aux.distance;
upDCp = top.aux.upDCp;
upDCn = top.aux.upDCn;
sum_upDCp = top.aux.sum_upDCp;
sum_upDCn = top.aux.sum_upDCn;

minus1s = -ones(size(pD));
if onGPU, minus1s = gpuArray(minus1s); end

% -----------------------------------------------------------------------------
% 1. H/P(D)
% -----------------------------------------------------------------------------
d_H_pD = deriv_ent(pD, minus1s);  % NxL

% -----------------------------------------------------------------------------
% 2. H/P(D|+), H/P(D|-) NXL
% -----------------------------------------------------------------------------
d_H_pDCp = diag(prCp) * d_H_pD;
d_H_pDCn = diag(prCn) * d_H_pD;

% -----------------------------------------------------------------------------
% 3. Hcond/P(D|+), Hcond/P(D|-) NXL
% ----------------------------------------------------------------------------- 
d_Hcond_pDCp = diag(prCp) * deriv_ent(pDCp, minus1s);
d_Hcond_pDCn = diag(prCn) * deriv_ent(pDCn, minus1s);

% -----------------------------------------------------------------------------
% 4. -MI/P(D|+), -MI/P(D|-): NXL
% -----------------------------------------------------------------------------
d_L_pDCp = -(d_H_pDCp - d_Hcond_pDCp);
d_L_pDCn = -(d_H_pDCn - d_Hcond_pDCn);

udlp = d_L_pDCp;
udln = d_L_pDCn;
% -----------------------------------------------------------------------------
% 4.1 -MI/uP(D|+), -MI/uP(D|-): NXL
% -----------------------------------------------------------------------------
if true
for i = 1:N
	d_L_pDCp(i, :) = ((repmat(-upDCp(i,:), L, 1)./(sum_upDCp(i)^2) + eye(L)*sum_upDCp(i)) ...
   						* d_L_pDCp(i, :)')';
	d_L_pDCn(i, :) = ((repmat(-upDCn(i,:), L, 1)./(sum_upDCn(i)^2) + eye(L)*sum_upDCn(i)) ...
   						* d_L_pDCn(i, :)')';
end
end
d_L_pDCp(isnan(d_L_pDCp)) = 0;
d_L_pDCn(isnan(d_L_pDCn)) = 0;
% -----------------------------------------------------------------------------
% 5. precompute dTPulse tensor
% -----------------------------------------------------------------------------
d_L_phi = zeros(nbits, N);
if onGPU, d_L_phi = gpuArray(d_L_phi); end
Np = sum(Xp, 2);
Nn = sum(Xn, 2);
invalid = (Np==0) | (Nn==0);

% new vectorization, loop over L instead of N
for l = 1:L
    % NxN matrix of delta_hat(i, j, l) for fixed l
    dpulse = triPulseDeriv(hdist, Cntrs(l), Delta);  % NxN
    ddp = dpulse .* Xp;  % delta_l^+(i, j)
    ddn = dpulse .* Xn;  % delta_l^-(i, j)

    % highly highly vectorized
    alpha_p = d_L_pDCp(:, l)./Np;  alpha_p(invalid) = 0;
    alpha_n = d_L_pDCn(:, l)./Nn;  alpha_n(invalid) = 0;
    alpha_p = diag(alpha_p);
    alpha_n = diag(alpha_n);
    Ap = ddp * alpha_p + alpha_p * ddp;  % 1st term: j=i, 2nd term: j~=i
    An = ddn * alpha_n + alpha_n * ddn;

    % accumulate gradient
    d_L_phi = d_L_phi - 0.5 * phi * (Ap + An);
end

% -----------------------------------------------------------------------------
% 6. -MI/x
% -----------------------------------------------------------------------------
sigmoid = (phi+1)/2;
d_phi_x = 2 .* sigmoid .* (1-sigmoid) * opts.sigmf_p(1);  % nbitsxN
d_L_x   = d_L_phi .* d_phi_x;

% -----------------------------------------------------------------------------
% 7. final
% -----------------------------------------------------------------------------
bot.dzdx = zeros(size(bot.x), 'single');
if onGPU, bot.dzdx = gpuArray(bot.dzdx); end
bot.dzdx(1, 1, :, :) = single(d_L_x);


% -----------------------------------------------------------------------------
% App. Update histogram parameters 
% -----------------------------------------------------------------------------

% Update Delta
d_L_Delta = 0;
dlr = 1e-4/N;
if true
	for l = 1:L
		dpulse = triPulseDeriv(hdist, Cntrs(l), Delta);
		ddp = sum(dpulse .* Xp, 2); % NX1
		ddn = sum(dpulse .* Xn, 2);
		d_L_Delta = d_L_Delta + d_L_pDCp(:, l)'*ddp + d_L_pDCn(:, l)'*ddn;
	end	
	Deltax = max(1, Deltax - dlr*d_L_Delta);
end
myLogInfo('%.2f', Deltax-dlr*d_L_Delta);
myLogInfo('%.2f', Deltax);
end

% -----------------------------------------------------------------------------
% derivative of entropy
% -----------------------------------------------------------------------------
function dHp = deriv_ent(p, minus1s)
dHp = minus1s;
dHp(p>0) = dHp(p>0) - log2(p(p>0));
end
