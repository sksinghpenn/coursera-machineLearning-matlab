function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% get hthetaX for layer 1

X = [ones(m, 1) X] % add theta0 with x0  =1
thetaTransposeX = X*Theta1';
% create inline function
hThetaXFunc = @(thetaTransposeX) 1 / (1 + exp(-thetaTransposeX));
% call in using vectorize programming paradigm
hThetaXLayer1 = arrayfun(hThetaXFunc,thetaTransposeX);

      



% get hthetaX for layer 2
hThetaXLayer1 = [ones(m, 1) hThetaXLayer1]
thetaTransposeX = hThetaXLayer1*Theta2';
% create inline function
hThetaXFunc = @(thetaTransposeX) 1 / (1 + exp(-thetaTransposeX));
% call in using vectorize programming paradigm
hThetaX = arrayfun(hThetaXFunc,thetaTransposeX);

% final output labels
A = ( hThetaX>= 0.5);
[X,p] = max(hThetaX, [], 2);

% =========================================================================


end
