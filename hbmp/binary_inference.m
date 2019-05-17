function [M, bit_weights, residual] = binary_inference(S, max_iter, tolerance, ...
												 weighted, regress, nbits)
% Please cite these papers if you use this code.
%
% 1. "Hashing with Binary Matrix Pursuit", 
%    Fatih Cakir, Kun He, Stan Sclaroff
%    European Conference on Computer Vision (ECCV) 2018
%    arXiV:1808.01990 
%
% 2. "Hashing with Mutual Information", 
%    Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
% 	 IEEE TPAMI 2019 (to appear)
%    arXiv:1803.00974
%
% 3. "MIHash: Online Hashing with Mutual Information", 
%    Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
%    International Conference on Computer Vision (ICCV) 2017
%    (* equal contribution)
% 
% INPUTS:
% 		S           : (2D matrix, float) Affinity matrix. Higher values means ...
%                     more similar 
% 		max_iter    : (scalar, integer) Maximum iteration
% 		tolerance 	: (scalar, float) tolerance value
% 		weighted 	: (boolean) weight each bit
% 		regress 	: (boolean) do regression after each step
% 
% OUTPUTS:
% 		M  			: (matrix, logical) Contains binary codes
% 		bit_weights : (vector, float) Contains weights for each bit
% 		residual 	: (scalar, float) Contains the final residual norm
%
%

% -----------------------------------------------------------------------------
% Input checking
% -----------------------------------------------------------------------------
if nargin < 5, regress = true; 		end;
if nargin < 4, weighted = true; 	end;
if nargin < 3, tolerance = 1e-6; 	end;
if nargin < 2, max_iter = 1e5; 		end;
if nargin < 1, error('No inputs.');	end;

% will fit binary codes with {-1, 1} values
S = (2*S - 1);	
bit_weights = [];
bit_weights_orig = [];

% a 'mask' matrix to mask self-similarity values during binary inference
Mask = ones(size(S)) + diag(-ones(1, size(S, 1)));
% -----------------------------------------------------------------------------
% Binary inference with unweighted bits
% -----------------------------------------------------------------------------
if ~weighted

	% scale to max_iter 
	S = S * max_iter;

	% greedy unweigted fitting
	for t = 1:max_iter
		if t > 1
			y = 2*single(y) - 1;
			S = S - y*y';
		end	
		% eigendecomposition
		[U, V] = eig(S);
		eigenvalue = diag(V)';
		% get the eigenvector with the highest eigenvalue
		[eigenvalue, order] = sort(eigenvalue, 'descend');
		y = U(:, order(1));
		y = y > 0;
		M(:, t) = y;
	end
    y = 2*single(y)-1;
	S = S - y*y';
	residual = norm(Mask.*S,'fro');
	myLogInfo('Residual = %g', residual);
	return
end		

% -----------------------------------------------------------------------------
% Binary inference with weighted bits
% -----------------------------------------------------------------------------
% count the times the gradient direction is near zero
near_zero = 0;

% lipchitz constant
L = norm(Mask.*S,'fro'); 

X = zeros(size(S));

% residual matrix
R = S;

% if regress is set, get the non-masked entries -for now this always corresponds
% to non-diagonal entries
if regress
	m_i = find(Mask(:) == 1);
	Y = S(:);
	Y = Y(m_i);
end


fprintf('inferring bit\n');
% do binary inference
for t = 1:max_iter+1
   	fprintf('%d ', t);	
	if norm(Mask.*(X-S), 'fro') < tolerance && t > nbits + 1
		myLogInfo('Early break at %.5d\n', t);
		break;
	end	
	% keep track of residual norm
	B(t) = norm(Mask.*-R,'fro');
	
	if t > 1
		% set bit weights, D holds the negative gradient norm
		bit_weights(t) = D(t-1)/(L*length(y)^2);
		bit_weights_orig(t) = bit_weights(t);
		% add binary code to output matrix
		M(:, t) = logical((y + 1)./2);
		
		% track the sum of the weighted rank-one matrices
		X = X + bit_weights(t) * (y*y');
			
		if regress
			% for regression hold the rank-one matrices
			% notice that t=0 is a zero matrix
			YM(:,:,t) = y*y';
			% convert matrices into columns
			c_X = zeros(length(m_i), t-1);
			for i = 2:t
				% columnize
				T = YM(:,:,i);
				c_X(:, i-1) = T(m_i);
			end
		if true
			% do regression & get optimal weights
			t_bit_weights = (pinv(c_X)*Y)';
		else % alternative constrain the bit weights to positive values
			myLogInfo('Constraining to positive coefficients');
			options = optimoptions('lsqlin','Algorithm','interior-point', 'Display', 'off');
			t_bit_weights = lsqlin(double(c_X), double(Y), -eye(size(c_X, 2)), zeros(1, size(c_X,2)), [], [], [], [], [],  options);
		end

		% sum of weighted rank-one matrices where the weights are regressed
		t_X = sum(bsxfun(@times, YM(:,:,2:end), reshape(t_bit_weights,1,1,[])), 3);

		% check whether the objective has decreased
		% due to numerical precision issues this might not be the case
		if norm(Mask.*(X-S),'fro') > norm(Mask.*(t_X-S), 'fro')
			bit_weights(2:end) = t_bit_weights;
			X = t_X;
        end
        end
    end
	% get residual matrix
    R = (X - S);
	
	% eigendecomposition
    [U, V] = eig(single(Mask.*-R));
    
	% get the eigenvector with the highest eigenvalue
	eigenvalue = diag(V)';
    [eigenvalue, order] = sort(eigenvalue, 'descend');
    y = U(:, order(1));

	% binarize
    y = y > 0;

	% convert to {-1, 1} for further computation
	y = 2*single(y) - 1; 
	
	% if the below condition is true then the gradient direction is positive
	% (this might happen due to binarization)
	% compute a new binary code
	% here we simply randomly generate on, although in practice there are 
	% much better ways to accomplish this
	if (t > 1 && any((2*single(M)-1)'*y == length(y))) || y'*(Mask.*-R)*y < 0 
		near_zero = near_zero + 1;
		break_ = true;
		while (t > 1 && any((2*single(M)-1)'*y == length(y))) || y'*(Mask.*-R)*y < 0
			y = 2*randi([0 1], length(y), 1)-1; 
        end
    end

	% check NaN's in weight vector
    if any(isnan(bit_weights)), error('Computational error, bit_weights is nan!'); end;

    % check if norm is monotonically decreasing
    assert(all(diff(B) <= 1e-4)); % considering numerical errors 

	% store the (negative) gradient norm
    D(t) = y'*(Mask.*-R)*y;
end

% remove first entries
M(:, 1) = [];
bit_weights_orig = bit_weights_orig(2:end)
bit_weights = bit_weights(2:end)
residual = B(end);
myLogInfo('Residual=%g, Avoided %d near nullspace solutions', ...
		residual, near_zero);

end

