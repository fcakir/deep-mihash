function res = evaluate_tieap(Htest, Htrain, Aff, opts, cutoff, bit_weights)
% input: 
%   Htrain - (logical) training binary codes
%   Htest  - (logical) testing binary codes
%   Aff    - Ntest x Ntrain affinity matrix

[nbits, Ntest] = size(Htest);
if isfield(opts, 'nbits'), assert(nbits == opts.nbits); end
assert(size(Htrain, 1) == nbits);
Ntrain = size(Htrain, 2);

phi_t = 2*Htest  - 1;
phi_r = 2*Htrain - 1; 
hdist = (nbits - phi_t' * phi_r)/2;  % pairwise dist matrix
Aff   = Aff > 0;
Np    = sum(Aff, 2);

fastAP = zeros(1, Ntest);
LB_APt = zeros(1, Ntest);
APt    = zeros(1, Ntest);
t0 = tic;
for i = 1:Ntest
    if Np(i) == 0, continue; end

    nD  = accumarray(hdist(i, :)'+1, 1, [nbits+1, 1]);
    nDp = accumarray(hdist(i, Aff(i, :))'+1, 1, [nbits+1, 1]);
    NDp = cumsum(nDp);
    ND  = cumsum(nD);

    Np0 = zeros(size(NDp));
    N0  = zeros(size(ND ));
    Np0(2:end) = NDp(1:end-1);
    N0 (2:end) = ND (1:end-1);

    % 1. compute FastAP
    fastAP_i = nDp .* NDp ./ ND;
    fastAP_i(ND == 0) = 0;
    fastAP(i) = sum(fastAP_i) / Np(i);

    % 2. compute AP LB
    %
    % 2.1 copy FastAP
    LB_APt_i = fastAP_i;
    %
    % 2.2 monoticity test
    rhs = (Np0 + 1) ./ (N0 + 1);
    mul = (nDp - 1) ./ (nD - 1);
    %
    % 2.2.1 APLB: when increasing in j, C is concave in j, simply use C(d, 1)
    ind = (nD>1) & mul>rhs;
    LB_APt_i(ind) = nDp(ind) .* rhs(ind);
    %
    % 2.2.2 APLB: when decreasing in j, C actually CONVEX in j
    %       use Jensen's inequality, use C(d, (n_d+1)/2)
    ind = (nD>1) & mul<=rhs;
    C = (Np0 + ((nD+1)/2-1).*mul + 1) ./ (N0 + (nD+1)/2);
    LB_APt_i(ind) = nDp(ind) .* C(ind);
    %
    % 2.3 divide by N+
    LB_APt(i) = sum(LB_APt_i) / Np(i);

    % 3. compute AP_t, only those that differ from fastAP
    APt_i = fastAP_i;
    for l = find(nD>1 & nDp>0)'
        mult = mul(l); %(nDp(l) - 1) / (nD(l) - 1);
        nume = Np0(l) + 1 + (0:nD(l)-1) * mult;
        deno = N0(l) + (1:nD(l));
        APt_i(l) = mean(nume./deno) * nDp(l);
    end
    AP_t(i) = sum(APt_i) / Np(i);
end
%myLogInfo(' FastAP = %g', mean(fastAP));
%myLogInfo(' LB_APt = %g', mean(LB_APt));
myLogInfo('AP-T = %g', mean(AP_t));
myLogInfo('%.2f seconds\n', toc(t0));
res = mean(AP_t);
end
