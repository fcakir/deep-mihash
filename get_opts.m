function opts = get_opts(dataset, nbits, modelType, varargin)
% Manages experimental parameters (opts) by creating an input parser object.
% The parameter values are initialized with the arguments passed in with the 
% 'demo_AP.m' function. See EXAMPLE COMMANDS in demo_AP.m .  
%
% Please cite the below papers if you use this code.
%
% 1. "Hashing with Mutual Information", 
%    Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
%    arXiv:1803.00974 2018
%
% 2. "MIHash: Online Hashing with Mutual Information", 
%    Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
%    International Conference on Computer Vision (ICCV) 2017
%    (* equal contribution)
% 
% INPUTS
%   dataset  	- (string) in {'cifar', 'nus', 'imagenet', 'labelme'}
%   nbits    	- (int) length of binary code
%  	modelType	- (string) in {'fc1', 'vggf', 'vggf_ft', 'alexnet_ft'} (among others) corresponding
%			   	  to the models as defined under '+models' folder. 

%   sigmf 	 	- (2D vector). First element sets the 'sigmoid sharpness'. 
% 				  It corresponds to the 'a' value in MATLAB's sigmf function.
% 				  The second element is zero and is kept for future purposes.  
% 	nbins 	 	- (int) Number of bins to be used for the distance distributions. 
% 				- A good value is nbins=nbits/2.
% 	obj 	 	- (string) A prefix that determines the loss layer. If set to 'mi'
% 				  then the loss functions 'mi_forward.m' and 'mi_backward' will be
% 				  used. 

% 	normalize 	- (bool) Determines whether to apply normalization to the input features.
% 				  Only applicable when input is GIST, FC7 or other similar feature descriptors.
% 	solver 		- (string) The SGD solver. Default option is 'sgd'. 
%	batchSize 	- (int) Batch size.
% 	lr 			- (float) Learning rate
% 	lrstep 		- (int) Multiplies the learning rate with 'lrdecay' at every 'lrstep'.
% 	lrdecay 	- (float) Learning rate decay parameter
% 	
% 	lrmult 		- (float) Multiplies the learning rate for the pre-trained layers of
%				   alexnet_ft and vggf_ft. Must be between [0, 1]. 
% 	wdecay 		- (float) l2 regularization parameter. 
% 	bpdepth		- (int) Backpropagates the gradients to the last 'bpdepth' layers.
% 				  Set to Inf.
% 	dropout 	- (bool) Applies dropout after fc6 and fc7 layers. 

%	epoch 		- (int) Number of epochs. 
% 	gpus 		- (int vector) specificies the gpus id. If set to [], then cpu is
%				  used. The appropiate MatConvNet library must be installed. 
% 	continue 	- (bool) Whether to continue from last saved epoch. 
% 	debug 		- (bool) For debugging purposes.  
% 	split 		- (int) in {1, 2}. Applicable only for CIFAR and NUSWIDE. Specifies
% 				  the training/test set split See +imdb/split_* functions and the paper
% 				  for details.

%   metrics 	- (string) in {'AP', 'AP@5000', 'AP@50000'}. Evaluation metric.
%	testInterval- (int) Does evaluation at testInterval steps. 
% 	ep1			- (bool) Evaluate after first epoch?

% 	randseed	- (int) Random generator seed. 
% 	prefix 		- (string) Overrides the prefix in the experimental folder name, see 
% 				  process_opts.m .  
%
% OUTPUTS
%   opts 		- (struct) input parser results. Each field corresponds to a valid input 
% 		 		  argument. 
% 	

% -----------------------------------------------------------------------------
% Generic
% -----------------------------------------------------------------------------
ip = inputParser;
ip.addRequired('dataset'   , @isstr);
ip.addRequired('nbits'     , @isscalar);
ip.addRequired('modelType' , @isstr);

% -----------------------------------------------------------------------------
% model params
% -----------------------------------------------------------------------------
ip.addParameter('sigmf', [1 0]);  % 2nd para: =0 use fixed
ip.addParameter('nbins', nbits/2);  % use <nbits for less sparse histograms
ip.addParameter('obj'  , 'mi');

% -----------------------------------------------------------------------------
% feature params
% -----------------------------------------------------------------------------
ip.addParameter('normalize', true);

% -----------------------------------------------------------------------------
% SGD
% -----------------------------------------------------------------------------
ip.addParameter('solver'    , 'sgd');
ip.addParameter('batchSize' , 256);
ip.addParameter('lr'        , 0.1);
ip.addParameter('lrstep'    , 10);
ip.addParameter('lrdecay'   , 0.5);
ip.addParameter('lrmult'    , 0.01);
ip.addParameter('wdecay'    , 5e-4);
ip.addParameter('bpdepth'   , inf);
ip.addParameter('dropout'   , 0);

% -----------------------------------------------------------------------------
% train
% -----------------------------------------------------------------------------
ip.addParameter('epoch'    , 50);
ip.addParameter('gpus'     , []);
ip.addParameter('continue' , true);
ip.addParameter('debug'    , false);
ip.addParameter('split'    , 2);

% -----------------------------------------------------------------------------
% test
% -----------------------------------------------------------------------------
ip.addParameter('metrics', 'AP');
ip.addParameter('testInterval', 10);
ip.addParameter('ep1', false);

% -----------------------------------------------------------------------------
% misc
% -----------------------------------------------------------------------------
ip.addParameter('randseed', 0);
ip.addParameter('prefix', []);

% -----------------------------------------------------------------------------
% parse input
% -----------------------------------------------------------------------------
ip.KeepUnmatched = true;
ip.parse(dataset, nbits, modelType, varargin{:});
opts = ip.Results; 

end
