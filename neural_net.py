#!/home/zeus/miniconda2/envs/py2/bin/python
import torch
from torch import autograd,nn,optim
import torch.nn.functional as F

batchSize = 5
inputSize = 4
hiddenSize = 10
outputSize = 4
learningRate = 0.001
epochs = 5000

torch.manual_seed(99)
inputData = autograd.Variable(torch.rand(batchSize, inputSize))
target = autograd.Variable((torch.rand(batchSize)*outputSize).long())

class Net(nn.Module):
    def __init__(self,inputSize,hiddenSize, outputSize):
        super(Net,self).__init__()
        self.h1 = nn.Linear(inputSize,hiddenSize)
        self.h2 = nn.Linear(hiddenSize,outputSize)

    def forward(self,x):
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        x = F.log_softmax(x)
        return x


model = Net(inputSize=inputSize,hiddenSize=hiddenSize,outputSize=outputSize)
opt = optim.Adam(params=model.parameters(),lr=learningRate) #iterator of the parameters, registers with the Module, iterates when in optimizer


for epoch in range(epochs):
    out = model(inputData) #forward pass
    _,predictedData = out.max(1)
    loss = F.nll_loss(out,target) #calculate the loss

    _,predictedData = out.max(1)

    print('predicted data',str(predictedData.view(1,-1)).split('\n')[1])
    print('target data',str(target.view(1,-1)).split('\n')[1])
    print('loss',loss.data[0])

    model.zero_grad() #automatically make the gradients on the weights zero for backprop each time
    loss.backward()
    opt.step() #take the step
    if loss.data[0]<0.1:
        print('\ntrained successfully.\n')
        break

print('predicted data',str(predictedData.view(1,-1)).split('\n')[1])
print('target data',str(target.view(1,-1)).split('\n')[1])
print('loss',loss.data[0])
