addpath('util', 'batch-generator', 'encoder', 'getter', 'mihash', 'tester', ...
    'trainer');

[r, name] = unix('hostname');
name = name(1:end-1);
myLogInfo('Hostname = %s', name);
run ./matconvnet/matlab/vl_setupnn
myLogInfo('MatConvNet ready');
