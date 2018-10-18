function saveps = set_saveps(opts)
% -----------------------------------------------------------------------------
% set model save checkpoints:
% the model files are saved at every epoch value as specified by 'saveps' below. 
% -----------------------------------------------------------------------------
saveps = [0: opts.lrstep: opts.epoch, opts.epoch];
end
