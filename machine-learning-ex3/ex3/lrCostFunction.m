function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


thetaTransposeX = X*theta;


% create inline function
hThetaXFunc = @(thetaTransposeX) 1 / (1 + exp(-thetaTransposeX));


% call in using vectorize programming paradigm

hThetaX = arrayfun(hThetaXFunc,thetaTransposeX);


loghThetaX = log(hThetaX);


yLoghThetaX = y.*loghThetaX;
minusYLoghThetaX =  -1.*yLoghThetaX;

oneMinusY = 1 - y;

oneMinushThetaX = 1 - hThetaX;

logOneMinushThetaX = log(oneMinushThetaX);

oneMinusYlogOneMinushThetaX = oneMinusY.*logOneMinushThetaX;

minusYLoghThetaXMinusOneMinusYlogOneMinushThetaX = minusYLoghThetaX - oneMinusYlogOneMinushThetaX;


sigma = sum(minusYLoghThetaXMinusOneMinusYlogOneMinushThetaX);

% regularization related
thetaByExludingThetaZero = theta([2:length(theta)],:);
thetaSquare = thetaByExludingThetaZero.^2;

sumOfthetaSquare= sum(thetaSquare);


multipleByLambdaAndDivide= lambda.*sumOfthetaSquare;


J = sigma/m + multipleByLambdaAndDivide/(2*m);



%calculation of gradient

%disp('value of theta is'), disp(theta);
%disp('size of theta is'), disp(size(theta));

%calculation of gradient

hThetaXMinusY = hThetaX-y;

hThetaXMinusYMultiplyByX = hThetaXMinusY.*X;

% unreguralize gradient
unregularized_grad = sum(hThetaXMinusYMultiplyByX)/m;

% for the rest of the theta elements
lambdaDivMMultipliedByTheta = ((lambda/m).*theta)';
%disp('lambdaDivMMultipliedByTheta '), disp(lambdaDivMMultipliedByTheta)
%disp('size of lambdaDivMMultipliedByTheta'), disp(size(lambdaDivMMultipliedByTheta))
grad = sum(hThetaXMinusYMultiplyByX)/m + lambdaDivMMultipliedByTheta; 
%disp('size of grad '), disp(size(grad))

grad(1,1) = unregularized_grad(1,1);



grad = grad(:);

end
