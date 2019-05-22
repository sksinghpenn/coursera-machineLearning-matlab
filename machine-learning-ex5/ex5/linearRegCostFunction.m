function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

m = length(X);
xTheta = X*theta;
thetaZeroMinusThetaOneX1MinusY =xTheta - y;
J_without_regularized = sum(thetaZeroMinusThetaOneX1MinusY.^2)/(2*m); 

% regularization related
thetaByExludingThetaZero = theta([2:length(theta)],:);
thetaSquare = thetaByExludingThetaZero.^2;

sumOfthetaSquare= sum(thetaSquare);


multipleByLambdaAndDivide= lambda.*sumOfthetaSquare;


J = J_without_regularized + multipleByLambdaAndDivide/(2*m);


% =========================================================================

%calculation of gradient

hThetaXMinusY = xTheta-y;

hThetaXMinusYMultiplyByX = hThetaXMinusY.*X;

% unreguralize gradient
unregularized_grad = sum(hThetaXMinusYMultiplyByX)/m;

% for the rest of the theta elements
lambdaDivMMultipliedByTheta = ((lambda/m).*theta)';

grad = sum(hThetaXMinusYMultiplyByX)/m + lambdaDivMMultipliedByTheta; 
%disp('size of grad '), disp(size(grad))

grad(1,1) = unregularized_grad(1,1);



grad = grad(:);

end
