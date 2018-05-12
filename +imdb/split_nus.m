function S = nus_split(Y, opts)
% Sets the training set size. Two different partitionings are used: split=1 
% where 500 and 100 instances are samples from each class for training and test
% set. Split=2 samples 100 instances per class for test set. The remaining images
% are used for the training set. All non-test images are used as the retrieval set.
% Note that only 21 most frequent concepts are assumed to be used. 
% See paper for more details about the experimental setup. 
%
% INPUTS
%  	   Y 	 - (nxl) Each column is the concept membership indicator of an image.
%   opts 	 - (struct) options, see get_opt.m and process_opts.m . 
%
% OUTPUTS
%    set     - (nx1 vector) Each element is from {1,2,3} indicating 
% 						a training, validation or test instace.
%
if opts.split == 1
    trainPerCls = 500; testPerCls = 100;
else, assert(opts.split == 2);
    trainPerCls = 0; testPerCls = 100;
end

[N, L] = size(Y);  assert(L == 21);
S = 2 * ones(N, 1);   % default: val
chosen = false(N, 1);
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
% Fatih's bugfix
if opts.split == 2
    S(S == 2) = 1;
end
end
