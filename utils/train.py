import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#----------------------基于lenet的网络，可选激活函数和池化方式----------------------
def make_net(func_type = 'relu', pool_type = 'avgpool'):
    if func_type == 'relu':
        if pool_type == 'avgpool':
            net = nn.Sequential(nn.Conv2d(1, 6, 5, padding = 2),
                                nn.ReLU(),
                                nn.AvgPool2d(2, 2),
                    
    
                                nn.Conv2d(6, 16, 5),
                                nn.ReLU(),
                                nn.AvgPool2d(2, 2),
    
                                nn.Flatten(),
                                nn.Linear(16 * 5 * 5, 120),
                                nn.ReLU(),
                                nn.Linear(120, 84),
                                nn.ReLU(),
                                nn.Linear(84, 10))
        else:
            net = nn.Sequential(nn.Conv2d(1, 6, 5, padding = 2),
                                nn.ReLU(),
                                nn.MaxPool2d(2, 2),
                    
    
                                nn.Conv2d(6, 16, 5),
                                nn.ReLU(),
                                nn.MaxPool2d(2, 2),
    
                                nn.Flatten(),
                                nn.Linear(16 * 5 * 5, 120),
                                nn.ReLU(),
                                nn.Linear(120, 84),
                                nn.ReLU(),
                                nn.Linear(84, 10))
    else:
        if pool_type == 'avgpool':
            net = nn.Sequential(nn.Conv2d(1, 6, 5, padding = 2),
                                nn.Sigmoid(),
                                nn.AvgPool2d(2, 2),
                    
    
                                nn.Conv2d(6, 16, 5),
                                nn.Sigmoid(),
                                nn.AvgPool2d(2, 2),
    
                                nn.Flatten(),
                                nn.Linear(16 * 5 * 5, 120),
                                nn.Sigmoid(),
                                nn.Linear(120, 84),
                                nn.Sigmoid(),
                                nn.Linear(84, 10))
        else:
            net = nn.Sequential(nn.Conv2d(1, 6, 5, padding = 2),
                                nn.Sigmoid(),
                                nn.MaxPool2d(2, 2),
                    
    
                                nn.Conv2d(6, 16, 5),
                                nn.Sigmoid(),
                                nn.MaxPool2d(2, 2),
    
                                nn.Flatten(),
                                nn.Linear(16 * 5 * 5, 120),
                                nn.Sigmoid(),
                                nn.Linear(120, 84),
                                nn.Sigmoid(),
                                nn.Linear(84, 10))
    return net
#---------------------------------------------------------------------------

#------------------------------------loss选取------------------------------
LOSSES = {
    'mse':         nn.MSELoss,
    'l1':          nn.L1Loss,
    'huber':       nn.SmoothL1Loss,
    'bce':         nn.BCELoss,              # 需 Sigmoid 输出
    'bce_logits':  nn.BCEWithLogitsLoss,    # 直接吃 logits
    'ce':          nn.CrossEntropyLoss,
}

def get_loss(name):
    if name not in LOSSES:
        raise ValueError(f'未知 loss: {name}')
    return LOSSES[name]()
#-----------------------------------------------------------------------------

#----------------------训练主函数(返回测试/训练的acc与loss的list）----------------------
def train(net, train_iter, test_iter, lr = 0.01, num_epochs = 10, optim = 'SGD', loss = 'ce'):
    net.to(device)
    
    #trainer
    if optim == 'SGD':
        trainer = torch.optim.SGD(net.parameters(), lr = lr)
    else:
        trainer = torch.optim.Adam(net.parameters(), lr = lr)
    
    #main loop
    train_loss = []
    test_loss = []
    test_acc = []
    loss = get_loss(loss)
    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            epoch_loss += l.item()
        train_loss.append((epoch + 1, epoch_loss / len(train_iter)))

        testloss, test_accuracy = evaluate(net, test_iter, loss)
        test_loss.append((epoch + 1, testloss))
        test_acc.append((epoch + 1, test_accuracy))
                        
    return train_loss, test_loss, test_acc
#-------------------------------------evaluate-----------------------------
def evaluate(net, test_iter, loss):    #注意计算的是一个epoch的acc与loss
    total_sample = 0
    correct_sample = 0
    total_l = 0
    net.eval()
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)  #已返回该批次的平均loss，注意不要多除了
            total_l += l.item()
            correct_sample += (y_hat.argmax(dim = 1) == y).sum().item()
            total_sample += y.numel()

        avg_loss = total_l / len(test_iter)
        avg_acc = correct_sample / total_sample

    return avg_loss, avg_acc
#------------------------------------------------------------------------------