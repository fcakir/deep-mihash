function Aff = affinity_binary(Y1, Y2, X1, X2, opts)
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
% binary affinity, returns logical matrix
if opts.unsupervised
    assert(~isempty(X1) && ~isempty(X2));
    [N1, Dx]  = size(X1);
    [N2, Dx2] = size(X2); assert(Dx2 == Dx);
    raw_dist = bsxfun(@plus, sum(X1.^2, 2), sum(X2.^2, 2)') - 2*X1*X2';
    Aff = (raw_dist <= max(opts.thr_dist));
else
    assert(~isempty(Y1) && ~isempty(Y2));
    [N1, Dy] = size(Y1);
    [N2, Dy2] = size(Y2); assert(Dy2 == Dy);
    if Dy == 1
        Aff = bsxfun(@eq, Y1, Y2');
    else
        Aff = (Y1 * Y2' > 0);
    end
end
if strcmp(opts.obj, 'hbmp')
	Aff = single(Aff);
end
end
