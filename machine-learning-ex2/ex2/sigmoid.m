function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ============================================================
% Computes the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
% =============================================================

% create inline function
sigmoidFunc = @(z) 1 / (1 + exp(-z));

% call in using vectorize programming paradigm
g = arrayfun(sigmoidFunc,z);

end
