function my_nn = make_nn_fwd_pass(input,my_nn)

% function my_nn = make_nn_fwd_pass(input,my_nn)
%
% Makes forward pass through neural network
% Arguments:
%   input = input sample (column vector)
%   my_nn = cell array structure defining neural network
%
% Output in my_nn.a{Nlayer}
%
% (c) 2020 Andres Kwasinski

my_nn.a{1} = input;
for lay = 2:my_nn.Nlayer
   my_nn.z{lay} = my_nn.w{lay} *  my_nn.a{lay-1} + my_nn.b{lay};
   my_nn.a{lay} = my_nn.Act_Fun{my_nn.actFunIndx{lay}}(my_nn.z{lay});
end


        