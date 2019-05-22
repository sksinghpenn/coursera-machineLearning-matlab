function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %











    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

    %disp(fprintf("cost %f ", J_history(iter)));
    % hyphothesis is formed by multiplying X matrix and the theta vector
    % size of X = m x n
    % size of theta = n x 1
    h= X*theta;

    % error vector is difference between h and y
    err = h - y;

    % change in theta (gradient) is the sum of product of Transpose X and error
    % vector
    % you will transpose X to get the result of same size as theta
    theta_change = (alpha*(X'*err))/m;

    theta = theta - theta_change;

    %disp(fprintf("gradient %f ", theta));


end

end
