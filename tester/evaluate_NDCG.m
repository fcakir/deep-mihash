function res = evaluate_NDCG(Htest, Htrain, Aff, opts, cutoff, bit_weights)
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
% input: 
%   Htrain - (logical) training binary codes
%   Htest  - (logical) testing binary codes
%   Aff    - Ntest x Ntrain affinity matrix

[nbits, Ntest] = size(Htest);
if isfield(opts, 'nbits'), assert(nbits == opts.nbits); end
assert(size(Htrain, 1) == nbits);
Ntrain = size(Htrain, 2);
if isempty(cutoff)
    cutoff = Ntrain;
    myLogInfo('Eval full ranking');
else
    cutoff = min(cutoff, Ntrain);
    myLogInfo('Ranking cutoff = %d', cutoff);
end

t0 = tic;
Aff   = max(0, Aff);
DCGi = zeros(Ntest, 1);
DCGr = DCGi;
discount = 1 ./ log2((1:Ntrain) + 1);

sim = compare_hash_tables(Htrain, Htest, bit_weights);
for i=1:Ntest	
	if all(Aff(i,:) == 0)
		DCGi(i) = 1; DCGr(i) = 1; 
		continue;
	end
	G = 2.^(Aff(i,:)) - 1;
	% ideal DCG
	[Gi, Io] = sort(G, 'descend');  % ideal ordering
	DCGi(i) = dot(single(Gi(1:cutoff)), discount(1:cutoff));
	[~, ir] = sort(sim(:,i), 'descend'); % sim is from nbits (sim) -> -nbits (non-sim)
	G = single(G);
	DCGr(i) = dot(G(ir(1:cutoff)), discount(1:cutoff));
end	
DCGr = DCGr ./ DCGi;
myLogInfo('      NDCG = %g', mean(DCGr));
res = [mean(DCGr), 0, 0]

if false
	% compute the residual of the fitting matrix
	% NOTE: this assumes the maximum Aff value to be 1
	%   	and the binary inference is done on a scaled
	% 		similarity matrix, i.e., S = S*opts.nbits;
	assert(max(Aff(:)) < 1+eps);
	residual = zeros(Ntest, 1);
	for i=1:Ntest
		if all(Aff(i,:) == 0)
			residual(i) = NaN;
			continue;
		end
		ind = find(single(Aff(i, :)) < 1);
		A = 2*single(Aff(i,ind))-1;
		A = opts.nbits*A;
		residual(i) = norm(abs(A - sim(ind,i)'),'fro');
	end
	residual = residual(~isnan(residual));
	myLogInfo(' 	Residual = %g', mean(residual));
end
end
