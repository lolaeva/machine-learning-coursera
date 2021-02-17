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
% ------------------------- PART1 ---------------------------------------
% --------------- cost function --------------
% Expand the 'y' output values into a matrix of single values
% y is matrix of vectors containing only values 0 or 1
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:); % its size is m x num_labels=5000x10

a1 = [ones(m,1), X];            % 5000x401
z2 = a1 * Theta1';              % 5000x401 * 401x25 --> 5000x25
a2 = [ones(m,1), sigmoid(z2)];  % 5000x26
z3 = a2 * Theta2';              % 5000x26 * 26x10 --> 5000x10
a3 = sigmoid(z3);               % 5000x10
% it is necessary to do an elementwise multiplication
J = (1/m) * sum(sum( (-y_matrix) .* log(a3) - (1-y_matrix) .* log(1-a3) ));

% ----------- regularized cost function --------------
% exclude terms that correspond to the bias from weights
t1 = Theta1(:, 2:end);
t2 = Theta2(:, 2:end); %10x25
% calculate regularization terms 
reg = lambda/(2*m) * ( sum(sum(t1.^2)) + sum(sum(t2.^2)) );
J = J + reg;

% ---------------------------- PART 2 -------------------------------------
% --------------- backpropagation --------------
for i=1:m
    a1 = [1; X(i,:)'];         % 401x1
    z2 = Theta1 * a1;          % 25x401 * 401x1 --> 25x1
    a2 = [1; sigmoid(z2)];     % 26x1
    z3 = Theta2 * a2;          % 10x26 * 26x1 --> 10x1
    a3 = sigmoid(z3);        
    delta_3 = a3 - y_matrix(i,:)'; % 10x1
    %                    26x10 * 10x1 --> 26x1 .* 26x1 -->26x1
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z2]);
    delta_2 = delta_2(2:end); % remove delta_2[0] 25x1
    mult1 = delta_2 * a1';    % 25x1 * 1x401 --> 25x401
    mult2 = delta_3 * a2';    % 10x1 * 1x26 --> 10x26
    Theta1_grad = Theta1_grad + mult1;
    Theta2_grad = Theta2_grad + mult2;
end
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;
    
    
% --------------- regularized backpropagation --------------
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
