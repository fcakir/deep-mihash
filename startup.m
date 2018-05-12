addpath('util');

[r, name] = unix('hostname');
name = name(1:end-1);
myLogInfo('Hostname = %s', name);

try
    if ismember(name, {'ivcgpu3.bu.edu' 'hopper'})
        myLogInfo('Using MatConvNet-GPU');
        run ./matconvnet/matlab/vl_setupnn
    elseif ismember(name, {'ivcgpu1.bu.edu' 'ivcgpu2.bu.edu'})
        myLogInfo('Using MatConvNet-CuDNNv5');
        run ./matconvnet_cudnn5/matlab/vl_setupnn
    else
        myLogInfo('Using MatConvNet-CPU');
        run ./matconvnet_cpu/matlab/vl_setupnn
    end
catch
    myLogInfo('Actually, scratch that');
    run ./matconvnet_scc/matlab/vl_setupnn
    global onSCC
    onSCC = true;
    myLogInfo('We are on SCC!');
end
myLogInfo('MatConvNet ready');
