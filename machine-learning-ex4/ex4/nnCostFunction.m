function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y_matrix = [];
for i = 1:num_labels
    y_matrix = [y_matrix y == i];
end;

X = [ones(size(X,1),1) X];


%layer 1
thetaTransposeX = X*Theta1';
% create inline function
hThetaXFunc = @(thetaTransposeX) 1 / (1 + exp(-thetaTransposeX));
% call in using vectorize programming paradigm
hThetaXLayer1 = arrayfun(hThetaXFunc,thetaTransposeX);
hThetaXLayer1Derivative = sigmoidGradient(thetaTransposeX);


%layer 2
hThetaXLayer1 = [ones(size(hThetaXLayer1,1),1) hThetaXLayer1];
thetaTransposeX2 = hThetaXLayer1*Theta2';
% create inline function
hThetaXFunc = @(thetaTransposeX2) 1 / (1 + exp(-thetaTransposeX2));
% call in using vectorize programming paradigm
hThetaXLayer2 = arrayfun(hThetaXFunc,thetaTransposeX2);
    
    
 
   

    
    
sigma = zeros(size(X,1),size(num_labels,1))
for i = 1:num_labels
   
    loghThetaX = log(hThetaXLayer2(:,i))



    yLoghThetaX = y_matrix(:,i).*loghThetaX;
    minusYLoghThetaX =  -1.*yLoghThetaX;

   
    oneMinusY = 1 - y_matrix(:,i);

    oneMinushThetaX = 1 - hThetaXLayer2(:,i);

    logOneMinushThetaX = log(oneMinushThetaX);

   

    oneMinusYlogOneMinushThetaX = oneMinusY.*logOneMinushThetaX;

    minusYLoghThetaXMinusOneMinusYlogOneMinushThetaX = minusYLoghThetaX - oneMinusYlogOneMinushThetaX;

  
    sigma = sigma + minusYLoghThetaXMinusOneMinusYlogOneMinushThetaX;

end;
   


%Theta2 row j and k count
theta2j = size(Theta2,1);
theta2k = size(Theta2,2);
   
%Theta1 row j and j count
theta1j = size(Theta1,1);
theta1k = size(Theta1,2);

sumTheta2 = 0;

for j = 1:theta2j
   
  sumTheta2 =  sumTheta2 + sum((Theta2(j,2:theta2k)).^2);
 
end



sumTheta1 =0;
for j = 1:theta1j
   
  sumTheta1 =  sumTheta1 + sum((Theta1(j,2:theta1k)).^2);
 
end



J = (sum(sigma)/m) + ((sumTheta1 + sumTheta2)*(lambda/2))/m;



    
% Unroll gradients


a1 = X;
z2  =  X*Theta1';
a2 =  hThetaXLayer1;
z3  =  a2*Theta2';
a3 =  hThetaXLayer2;


delta3 = a3 - y_matrix;

theta2_ignoring_bias = Theta2(:,2:(size(Theta2,2)));
delta3_multiplied_by_theta2_ignoring_bias = delta3*theta2_ignoring_bias;
delta2 = delta3_multiplied_by_theta2_ignoring_bias.*hThetaXLayer1Derivative;

bigDelta2 =  a2'*delta3;
bigDelta1 =  a1'*delta2;

Theta1_grad = (bigDelta1/m)';
Theta2_grad = (bigDelta2/m)';


grad = [Theta1_grad(:) ; Theta2_grad(:)];




end
