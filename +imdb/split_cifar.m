function S = split_cifar(Y, opts)
% Performs the training-test set split. Two different splits are used: split=1 
% where 500 and 100 instances are sampled from each class for the training and test
% sets. Split=1 samples 1000 instances per class for the test set. The remaining images
% are used as the training set. All non-test images are used as the retrieval set.
% See paper for more details about the experimental setup. 
%
% INPUTS
%  	   Y 	 - (1x60000) Label vector for CIFAR-10 dataset.  
%   opts 	 - (struct) options, see get_opt.m and process_opts.m . 
%
% OUTPUTS
%    S       - (60000x1 vector) Each element is from {1,2,3} indicating 
% 						a training, validation and test instance, respectively.
%
if opts.split == 1
    trainPerCls = 500; testPerCls = 100;
else, assert(opts.split == 2);
    trainPerCls = 0; testPerCls = 1000;
end

if ~iscolumn(Y), Y = Y'; end
assert(size(Y, 1) == 60e3);
S = 2 * ones(size(Y, 1), 1);  % default = val(2)

for c = 1:10
    % use the first testPerCls for test, next trainPerCls for train
    % but if trainPerCls<=0, use the rest for train
    ind = find(Y == c);
    ind = ind(randperm(length(ind)));
    assert(length(ind) >= trainPerCls + testPerCls);

    itest = ind(1:testPerCls);
    if trainPerCls > 0
        itrain = ind(testPerCls+1:trainPerCls+testPerCls);
    else
        itrain = ind(testPerCls+1:end);
    end
    S(itest) = 3;
    S(itrain) = 1;
end
end
