function error_epoch = ex8_onemodel(x,t,ActHdLy,Nep,Nlayer,InitOp)

% function error_epoch = ex8_onemodel(x,t,ActHdLy,Nep,Nlayer,InitOp)
%
% Arguments:
%     x = input data
%     t = target data
%     ActHdLy = index to activation function in hidden layers
%     Nep = number of epochs
%     Nlayer -  Number of layers (counting input layer)
%     InitOp = 0 or 1, chooses between two options to initialize weights
%
% Output:
%     error_epoch = mean squared error for each training epoch.
%
% (c) 2020 Andres Kwasinski

Ninputs = size(x,1);

N_neuron_layer(1) = Ninputs; % "Neurons" in input layer
N_neuron_layer(Nlayer) = 1;  % Neurons in output layer
N_neuron_layer(2:Nlayer-1) = Ninputs*2;  % Neurons in each hidden layer
LearnRate = 0.0005;
mBatchSize = round(length(t)/10);
NmBatch = floor(length(t)/mBatchSize);

% Available activation functions (in order as entered):
% 1)ReLU, 2)linear, 3)saturated linear, 4)symmetric saturating linear,
% 5)Log sigmoid 6)Hyperbolic tangent sigmoid
Act_Fun = {@poslin,@purelin,@satlin,@satlins,@logsig,@tansig};
Act_Fun_Drvtv = {@(x)x>0 ; @(x)ones(size(x)) ; @(x)ones(size(x))-((x<0)|(x>=1)).*ones(size(x)) ; ...
                 @(x)ones(size(x))-((x<-1)|(x>=1)).*ones(size(x)) ; ... 
                 @(x)logsig(x).*(1-logsig(x)) ; @(x)tansig(x).^2};

my_nn.Nlayer = Nlayer;
my_nn.Act_Fun = Act_Fun;
my_nn.Act_Fun_Drvtv = Act_Fun_Drvtv;

% Initialize weights matrices,  activation and bias vectors
my_nn.a{1} = zeros( N_neuron_layer(1) , 1 );  
for lay = 2:Nlayer
    % One matrix for each layer after first one
    % 'lay' index in my_nn.w{lay} is for layer "to the right". Same for biases.
    % Row is "TO" and column is "FROM" connection
    if InitOp
        my_nn.w{lay} = 0.001*(rand( N_neuron_layer(lay) , N_neuron_layer(lay-1) ) - 0.5);  % Weights
    else
        my_nn.w{lay} = 20*(rand( N_neuron_layer(lay) , N_neuron_layer(lay-1) ) - 0.5);  % Weights
    end
    my_nn.b{lay} = zeros( N_neuron_layer(lay) , 1);  % Biases
    my_nn.a{lay} = zeros( N_neuron_layer(lay) , 1 );  % Outputs (after activation function)
end
    

% Define activation functions
my_nn.actFunIndx{1} = [];
for c = 2:Nlayer-1
    my_nn.actFunIndx{c} = ActHdLy;
end    
my_nn.actFunIndx{Nlayer} = 2;


TrainInput = x; % Each column is one sample
TrainTgt = t;

for ep = 1:Nep
    for mBCnt = 1:NmBatch
        % Initialize accumulators for minibatches weights/biases update
        for lay = 2:Nlayer
            AccW{lay} = zeros( N_neuron_layer(lay) , N_neuron_layer(lay-1) );  
            AccB{lay} = zeros( N_neuron_layer(lay) , 1);  
        end

        for mm = 1:mBatchSize
            input = TrainInput(:,mm+(mBCnt-1)*mBatchSize);
            trgt = TrainTgt(mm+(mBCnt-1)*mBatchSize);
 
            smpl_error(mm+(mBCnt-1)*mBatchSize) = norm(trgt-my_nn.a{Nlayer})^2;

        end
        % Update weights and biases, gradient descent
        for lay = 1:Nlayer
            my_nn.w{lay} = my_nn.w{lay} - LearnRate / mBatchSize * AccW{lay};
            my_nn.b{lay} = my_nn.b{lay} - LearnRate / mBatchSize * AccB{lay};
        end
    end
    
    % Track error
    error_epoch(ep) = mean(smpl_error);
end
    
  

