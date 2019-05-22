function cost = squaredErrorCost(A,b,x)


m = size(A,1); % number of training examples

Ax = A*x
sqlErrors = (Ax - b).^2; % sqaured error

cost =  sum(sqlErrors);
