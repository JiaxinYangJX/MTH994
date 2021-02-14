

import torch.nn as nn




class CNN(nn.Module):
    def __init__(self, filter1_n, filter2_n,
                filter1_size=[5,5], filter1_stride=1, filter1_padding=0,
                filter2_size=[3,3], filter2_stride=1, filter2_padding=0,
                pooling_size=[2,2], pooling_stride=2, bias_label=True):
        '''
        :param filter1_n
        '''
        super(CNN, self).__init__()
        self.filter1_n       = filter1_n
        self.filter2_n       = filter2_n
        self.filter1_size    = filter1_size
        self.filter1_stride  = filter1_stride
        self.filter1_padding = filter1_padding
        self.filter2_size    = filter2_size
        self.filter2_stride  = filter2_stride
        self.filter2_padding = filter2_padding
        self.pooling_size    = pooling_size
        self.pooling_stride  = pooling_stride
        self.bias_label      = bias_label

        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=filter1_n, kernel_size=filter1_size,
                    stride=filter1_stride, padding=filter1_padding, bias=bias_label),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_stride),
            
            nn.Conv2d(in_channels=filter1_n, out_channels=filter2_n, kernel_size=filter2_size,
                    stride=filter2_stride, padding=filter2_padding, bias=bias_label),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_stride),
        )

        self.Fc = nn.Sequential(
            nn.Linear(),
            nn.Softmax()
        )


    def forward(self,x):
        x = self.Conv(x)
        return x