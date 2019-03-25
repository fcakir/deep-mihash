function S = split_nus(Y, opts)
% Does the training-testing set split. Two different splits are used: split=1 
% where 500 and 100 instances are sampled from each class for training and test
% set. Split=2 samples 100 instances per class for test set. The remaining images
% are used for the training set. All non-test images are used as the retrieval set.
% Note that only 21 most frequent concepts are considered. 
% See paper for more details about the experimental setup. 
%
% INPUTS
%  	   Y 	 - (nxl) Each row is an image, columns are the concepts. 
%   opts 	 - (struct) options, see get_opt.m and process_opts.m . 
%
% OUTPUTS
%    S       - (nx1 vector) Each element is from {1,2,3} indicating 
% 						a training, validation or test instance, respectively.
%
if opts.split == 1
    trainPerCls = 500; testPerCls = 100;
else, assert(opts.split == 2);
    trainPerCls = 0; testPerCls = 100;
end

[N, L] = size(Y);  assert(L == 21);
S = 2 * ones(N, 1);   % default: val
chosen = false(N, 1);

% Assumes 21 most frequent concepts are used only.
for c = 1:21
    % use the first testPerCls for test, next trainPerCls for train
    % but if trainPerCls<=0, use the rest for train
    ind = find(Y(:,c)>0 & ~chosen);
    ind = ind(randperm(length(ind)));
    assert(length(ind) >= trainPerCls + testPerCls);

    itest = ind(1:testPerCls);
    if trainPerCls > 0
        itrain = ind(testPerCls+1:trainPerCls+testPerCls);
    else
        itrain = [];
    end
    S(itest) = 3;
    S(itrain) = 1;
    chosen([itest; itrain]) = true;
end

if opts.split == 2
    S(S == 2) = 1;
end
end
