close all ; clear all;
%Download hwk7files_forMycourses.zip and place them into an appropriate
%hwk7 directory.  
%Update the two paths below for your machine
addpath C:\Users\tbrad\Documents\MATLAB\CMPE677HW6\libsvm-3.18\windows
 
% We will use mnist hand written digits, '0' through '9'
load('ex4data1.mat');  %5000 Mnist digits from Andrew Ng class
n = size(X, 1);  %number of samples = 5000, 500 from each class
D = size(X,2);  %number of dimensions/sample.  20x20=400
C = length(unique(y));  %number of classes, class values are 1...10, where 10 is a digit '0'
 
% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
 
displayData(X(sel, :));  %This function is from Andrew Ng's online class
 
%Convert X and y data into Matlab nnet format:
inputs = X';
%one-hot encoding ground truth values 
targets = zeros(C,n);
for ii=1:n
        targets(y(ii),ii) =  1;
end
%If given one-hot encoding, can convert back to vector of ground truth
%class values:
% target1Dvector=zeros(n,1);
% for ii=1:n
%         target1Dvector(ii) = find(targets(:,ii) == 1);
% end
% max(target1Dvector-y) %this will be zero

fprintf('\nLoading Saved Neural Network Parameters ...\n')
 
% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');  %Pre-learned weights from Andrew Ng class
 
% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

% Weight regularization parameter (we set this to 0 here).
lambda = 0;
input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf(['Cost at parameters (no regularization): %f \n'], J);

% Weight regularization parameter (we set this to 1 here).
lambda = 1;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
fprintf(['Cost at parameters (with regularization): %f \n'], J);

%--------------------------------------------------------------------------
fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
 % Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
 
fprintf('\nTraining Neural Network... \n')
 
%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);
 
%  You can try different values of lambda, but keep lambda=1 for this exercise
lambda = 1;
 
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                    input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
 
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
 
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
 
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
 
% Visual weights- can uncomment out next two lines to see weights            
% fprintf('\nVisualizing Neural Network... \n')
% displayData(Theta1(:, 2:end));
 
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);



