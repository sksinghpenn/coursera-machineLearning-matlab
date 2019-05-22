function J = costFunctionJ(X, y, theta)

% X is the "design matrix" containing our training examples.
% y is the class labels

m = size(X,1); % number of training examples

disp(m)
predictions = X*theta; %predictions of hypothesis on all m examples
disp(predictions)
sqlErrors = (predictions - y).^2; % sqaured error
disp(sqlErrors)

J = 1/(2*m) * sum(sqlErrors);
