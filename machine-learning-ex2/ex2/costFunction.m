function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%if size(y,1) ~= size(X,1),
 %   y = y(size(X,1))

%disp('size of y');
%disp(size(y));
%disp('size of x');
%disp(size(X));

%disp(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%disp('size of X')
%disp(size(X));

thetaTransposeX = X*theta;

%disp('size of thetaTransposeX')
%disp(size(thetaTransposeX));

%disp(thetaTransposeX)

% create inline function
hThetaXFunc = @(thetaTransposeX) 1 / (1 + exp(-thetaTransposeX));


% call in using vectorize programming paradigm

hThetaX = arrayfun(hThetaXFunc,thetaTransposeX);


%disp(hThetaX);

% =============================================================
loghThetaX = log(hThetaX);


%disp('size of loghThetaX')
%disp(size(loghThetaX));

yLoghThetaX = y.*loghThetaX;
minusYLoghThetaX =  -1.*yLoghThetaX;

oneMinusY = 1 - y;

oneMinushThetaX = 1 - hThetaX;

logOneMinushThetaX = log(oneMinushThetaX);

%disp(logOneMinushThetaX)

oneMinusYlogOneMinushThetaX = oneMinusY.*logOneMinushThetaX;

minusYLoghThetaXMinusOneMinusYlogOneMinushThetaX = minusYLoghThetaX - oneMinusYlogOneMinushThetaX;

%disp(minusYLoghThetaXMinusOneMinusYlogOneMinushThetaX);
sigma = sum(minusYLoghThetaXMinusOneMinusYlogOneMinushThetaX);

J = sigma/m;

%calculation of gradient

hThetaXMinusY = hThetaX-y;

hThetaXMinusYMultiplyByX = hThetaXMinusY.*X;

grad = sum(hThetaXMinusYMultiplyByX)/m


end




