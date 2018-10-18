function lrvec = set_lr(opts)
% -----------------------------------------------------------------------------
% set learning rate
% -----------------------------------------------------------------------------
if opts.lrdecay>0 && opts.lrdecay<1
    cur_lr = opts.lr;
    lrvec = [];
    while length(lrvec) < opts.epoch
        lrvec = [lrvec, ones(1, opts.lrstep)*cur_lr];
        cur_lr = cur_lr * opts.lrdecay;
    end
else
    lrvec = opts.lr;
end
end
