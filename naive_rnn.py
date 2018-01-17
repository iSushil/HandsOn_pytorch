#!/home/zeus/miniconda2/envs/py2/bin/python

'''
    recurrent neural network implementaion in pytorch
'''
import torch
from torch import autograd

batch_size = 5
input_size = 5
seq_len = 8
hidden_size = 32
num_layers = 4

input = autograd.Variable(torch.rand(seq_len,batch_size,input_size))
print('input.size',input.size())
state = autograd.Variable(torch.zeros(num_layers,batch_size,hidden_size))

rnn = torch.nn.RNN(
    input_size = input_size,
    hidden_size = hidden_size,
    num_layers = num_layers,
    nonlinearity = 'tanh'
)

print('rnn',rnn)
out,state = rnn(input,state)
print('out.size():',out.size())
print('state.size:',state.size())
