function batchFunc = get_batchFunc(imdb, opts, net)
if ismember(opts.modelType, {'fc', 'fc1'})
    batchFunc = @batch_fc7;
else
    % models pre-trained on imagenet
    imgSize = opts.imageSize;
    meanImage = single(net.meta.normalization.averageImage);
    if isequal(size(meanImage), [1 1 3])
        meanImage = repmat(meanImage, [imgSize imgSize]);
    else
        assert(isequal(size(meanImage), [imgSize imgSize 3]));
    end
    batchFunc = @(I, B) batch_imagenet(I, B, imgSize, meanImage);
end
end
