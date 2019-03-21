function [rex, data] = cnn_encode_unsup(net, batchFunc, imdb, ids, onGPU, layerOffset)
[data, labels] = batchFunc(imdb, ids);
net.layers{end}.class = labels;
if onGPU
    data = gpuArray(data); 
    res = vl_simplenn(net, data, [], [], 'mode', 'test');
    rex = squeeze(gather(res(end-layerOffset).x));
else
    res = vl_simplenn(net, data, [], [], 'mode', 'test');
    rex = squeeze(res(end-layerOffset).x);
end
end
