function [AccW,AccB] = bckprop_mse_updtw(input,trgt,my_nn,AccW,AccB)

% function [AccW,AccB] = bckprop_mse_updtw(input,trgt,my_nn,AccW,AccB)
%
% Backpropagation training for one sample assuming MSE cost
% Arguments:
%   input = input sample (column vector)
%   trgt = target (column vector)
%   my_nn = cell array structure defining neural network
%   AccW = accumulated cost gradient for weights
%   AccB = accumulated cost gradient for bias
%
% Output in my_nn.a{Nlayer}
%   AccW = accumulated cost gradient for weights
%   AccB = accumulated cost gradient for bias
%
% (c) 2020 Andres Kwasinski


Nlayer = my_nn.Nlayer;

delta_onesmpl{Nlayer} = ( my_nn.a{Nlayer} - trgt ) .* my_nn.Act_Fun_Drvtv{my_nn.actFunIndx{Nlayer}}(my_nn.z{Nlayer});
AccW{Nlayer} = AccW{Nlayer} + delta_onesmpl{Nlayer} * my_nn.a{Nlayer-1}';  
AccB{Nlayer} = AccB{Nlayer} + delta_onesmpl{Nlayer};  

for lay = Nlayer-1:-1:2
    delta_onesmpl{lay} = ( my_nn.w{lay+1}' * delta_onesmpl{lay+1} ) .* my_nn.Act_Fun_Drvtv{my_nn.actFunIndx{lay}}(my_nn.z{lay});
    AccW{lay} = AccW{lay} + delta_onesmpl{lay} * my_nn.a{lay-1}';  
    AccB{lay} = AccB{lay} + delta_onesmpl{lay};  
end
