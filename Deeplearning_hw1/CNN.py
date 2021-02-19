from __future__ import print_function
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



def parse_args():
    '''
    read parameters from command line
    '''
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--kernel_1_n', type=int, default=5, 
                        help='number of kernels in the first convolutional layer')
    parser.add_argument('--kernel_2_n', type=int, default=5, 
                        help='number of kernels in the second convolutional layer')
    parser.add_argument('--fc', type=int, default=200, 
                        help='number of neurons in the fully connected layer')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')

    args = parser.parse_args()
    return args      



def read_dataset(feature_file, label_file):
    '''
    Read data set in *.csv to data frame in Pandas
    '''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values # convert values in dataframe to numpy array (features)
    y = df_y.values # convert values in dataframe to numpy array (label)
    return X, y


def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler #import libaray
    scaler = StandardScaler() # call an object function
    scaler.fit(X_train) # calculate mean, std in X_train
    X_train_norm1 = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm1 = scaler.transform(X_test) # we use the same normalization on X_test
    X_train_norm = np.reshape(X_train_norm1,(-1,1,28,28)) # reshape X to be a 4-D array
    X_test_norm = np.reshape(X_test_norm1,(-1,1,28,28))
    return X_train_norm, X_test_norm




class CNN_model(nn.Module):
    def __init__(self, d_input, d_output, kernel_1_n, kernel_2_n, fc_n,
                kernel_1_size=5, kernel_1_stride=1, kernel_1_padding=0,
                kernel_2_size=3, kernel_2_stride=1, kernel_2_padding=0,
                pooling_size=2, pooling_stride=2, bias_label=True):
        '''
        :param filter1_n
        '''
        super(CNN_model, self).__init__()
        self.d_input          = d_input # height of the image
        self.d_output         = d_output
        self.kernel_1_n       = kernel_1_n
        self.kernel_2_n       = kernel_2_n
        self.fc_n             = fc_n
        self.kernel_1_size    = kernel_1_size
        self.kernel_1_stride  = kernel_1_stride
        self.kernel_1_padding = kernel_1_padding
        self.kernel_2_size    = kernel_2_size
        self.kernel_2_stride  = kernel_2_stride
        self.kernel_2_padding = kernel_2_padding
        self.pooling_size     = pooling_size
        self.pooling_stride   = pooling_stride
        self.bias_label       = bias_label

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=kernel_1_n,
                                kernel_size=kernel_1_size, stride=kernel_1_stride,
                                padding=kernel_1_padding, bias=bias_label)
        nn.init.xavier_uniform_(self.conv1.weight)

        self.pool1 = nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_stride)

        self.conv2 = nn.Conv2d(in_channels=kernel_1_n, out_channels=kernel_2_n,
                                kernel_size=kernel_2_size, stride=kernel_2_stride,
                                padding=kernel_2_padding, bias=bias_label)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.pool2 = nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_stride)
        
        self.linear1 = nn.Linear(in_features=self.fc_dim(), out_features=fc_n,
                                bias=bias_label)
        nn.init.xavier_uniform_(self.linear1.weight)
        
        self.linear2 = nn.Linear(in_features=fc_n, out_features=d_output,
                                bias=bias_label)
        nn.init.xavier_uniform_(self.linear2.weight)

        self.relu    = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self,X):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        X = self.conv1(X)
        X = self.relu(X)
        X = self.pool1(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.pool2(X)
        
        # flatten
        X = X.view(X.size(0),-1)

        # fc
        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)
        y_hat = F.log_softmax(X, dim=1)
        #y_hat = self.softmax(X)
        return y_hat

    
    def fc_dim(self):
        '''
        calculate the input dimension for the first fully connected layer
        CNN:  d_out = 1 + (d_in - kernel_size + 2*padding) / stride
        pool: d_out = 1 + (d_in - kernel_size) / stride
        '''
        d_out = 1 + (self.d_input - self.kernel_1_size + 2*self.kernel_1_padding) // self.kernel_1_stride
        d_out = 1 + (d_out - self.pooling_size) // self.pooling_stride
        d_out = 1 + (d_out - self.kernel_2_size + 2*self.kernel_2_padding) // self.kernel_2_stride
        d_out = 1 + (d_out - self.pooling_size) // self.pooling_stride

        return d_out**2 * self.kernel_2_n


#=================================Training & Testing============================
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print('Train Epoch: {} [{}/{} ]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))


def test(model, device, epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if epoch % 2 == 0:
        print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))




def main():
    args = parse_args()
    X_train, y_train = read_dataset('./data/MNIST_X_train.csv', './data/MNIST_y_train.csv')
    X_test, y_test = read_dataset('./data/MNIST_X_test.csv', './data/MNIST_y_test.csv')
    X_train, X_test = normalize_features(X_train, X_test)

    #==================================Pack Data================================
    train_data = torch.from_numpy(X_train).float()
    test_data = torch.from_numpy(X_test).float()

    trainset = TensorDataset(train_data, torch.from_numpy(y_train.ravel()))
    testset = TensorDataset(test_data, torch.from_numpy(y_test.ravel()))

    # Define data loader
    train_loader = DataLoader(dataset=trainset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=X_test.shape[0], shuffle=False)

    #=================================Design Net================================
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)

    model = CNN_model(d_input=X_train.shape[2], d_output=10,
                    kernel_1_n=args.kernel_1_n,
                    kernel_2_n=args.kernel_2_n,
                    fc_n=args.fc)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_adjust = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5, last_epoch = -1)

    for epoch in range(1, args.epochs + 1):
        lr_adjust.step()
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, epoch, test_loader)


if __name__ == '__main__':
    main()
