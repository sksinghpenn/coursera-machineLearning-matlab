function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

% =============================================================

end
