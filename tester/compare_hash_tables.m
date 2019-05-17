function sim = compare_hash_tables(Htrain, Htest, bit_weights)
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
trainsize = size(Htrain, 2);
testsize  = size(Htest, 2);
if isempty(bit_weights)
	sim = (2*single(Htrain)-1)'*(2*single(Htest)-1);
else
if isrow(bit_weights), bit_weights = bit_weights'; end;
	Htrain = repmat(bit_weights, 1, trainsize) .* (2*single(Htrain)-1);
	Htest = (2*single(Htest)-1);
	sim = Htrain'*Htest;
end
end

