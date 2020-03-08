function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g =  1./(1+exp(-z));

% 1/(1+exp(-z)) without the dot operator gives a 1x10 row vector
% but we need a 10X1 vector so use the dot operator for element wise division

% =============================================================

end
