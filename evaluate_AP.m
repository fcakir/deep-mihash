function res = evaluate_AP(Htest, Htrain, Aff, opts, cutoff)
% Given binary codes for a query (Htest) and retrieval set (Htrain),
% computes the Mean Average Precision (at a certain cutoff, if provided).
% 
% Please cite the below papers if you use this code.
%
% 1. "Hashing with Mutual Information", 
%    Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
%    arXiv:1803.00974 2018
%
% 2. "MIHash: Online Hashing with Mutual Information", 
%    Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
%    International Conference on Computer Vision (ICCV) 2017
%    (* equal contribution)
%
% INPUTS
%   Htest    - (nbitsxNtest single) Hash table/binary codes for test set. 
%				See cnn_encode* functions. 
%   Htrain   - (nbitsxNtrain) Hash table/binary codes for test set. 
%				See cnn_encode* functions. 
%  	Aff      - (NtestxNtrain) logical affinity matrix. 
%   opts     - (struct) options, see get_opt.m and process_opts.m . 
%   cutoff   - (int) specifies the cutoff K of AP@K. If empty, full AP is computed. 
%
% OUTPUTS
%   res      - (float) Mean Average Precision
%
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

t0 	= tic;
Aff = Aff > 0;
APr = zeros(Ntest, 1);

phi_t = 2*Htest  - 1;
phi_r = 2*Htrain - 1; 
hdist = (nbits - phi_t' * phi_r)/2;  % pairwise dist matrix

for i = 1:Ntest
    A  = Aff(i, :);
    D  = hdist(i, :);
    ir = [];  % regular (tie-unaware)
    n  = 0;
    for d = 0 : nbits
        id = find(D == d);
        ir = [ir, id];
        n  = n + length(id);
        if n >= cutoff, break; end
    end
    APr(i) = get_AP( A(ir(1:cutoff)) );
end

myLogInfo('      AP = %g', mean(APr));
myLogInfo('%.2f seconds\n', toc(t0));
res = mean(APr) ;
end


function AP = get_AP(l)
    cl = cumsum(l);
    pl = cl ./ (1:length(l));
    if sum(l) ~= 0
        rl = cl ./ sum(l);
    else
        rl = zeros(1, length(cl));
    end
    drl = [0, diff(rl)];
    AP = sum(drl .* pl);            
end

