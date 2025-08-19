import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearSVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        outputs = torch.matmul(x, self.weights.t()) + self.bias
        return outputs

class RBFSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma):
        super(RBFSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))
        self.bn = nn.BatchNorm1d(input_size)  # Thêm BatchNorm1d

    def forward(self, x):
        # Apply BatchNorm only if batch_size > 1 and in training mode
        if x.size(0) > 1 and self.training:
            x = self.bn(x)
        
        # RBF kernel: K(x, w) = exp(-gamma * ||x - w||^2)
        # Expand dimensions for broadcasting: x [batch, 1, dim], weights [1, classes, dim]
        x_expanded = x.unsqueeze(1)  # [batch, 1, input_size]
        w_expanded = self.weights.unsqueeze(0)  # [1, num_classes, input_size]
        
        # Compute squared Euclidean distances
        dists_squared = torch.sum((x_expanded - w_expanded) ** 2, dim=2)  # [batch, num_classes]
        
        # Apply RBF kernel
        kernel_matrix = torch.exp(-self.gamma * dists_squared)
        outputs = kernel_matrix + self.bias
        return outputs

class PolySVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma, r, degree):
        super(PolySVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.degree = degree
        self.r = r
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        # dists = torch.cdist(x, self.weights, p=2)
        # kernel_matrix = (self.gamma * dists + self.r) ** self.degree #này sai công thức nhưng cho kết quả tốt?
        kernel_matrix = (self.gamma * torch.mm(x, self.weights.t()) + self.r) ** self.degree
        outputs = kernel_matrix + self.bias
        return outputs
    

class SigmoidSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma, r):
        super(SigmoidSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.r = r
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))
        self.bn = nn.BatchNorm1d(input_size)  # Thêm BatchNorm1d

    def forward(self, x):
        '       # Apply BatchNorm only if batch_size > 1 and in training mode
        if x.size(0) > 1 and self.training:
            x = self.bn(x)
            
        kernel_matrix = torch.tanh(self.gamma * torch.mm(x, self.weights.t())+ self.r)
        outputs = kernel_matrix  + self.bias
        return outputs


class CustomSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma, r, degree):
        super(CustomSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.degree = degree
        self.r = r
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        dists = torch.cdist(x, self.weights, p=2)
        kernel_matrix = (self.gamma * dists + self.r) ** self.degree #này sai công thức nhưng cho kết quả tốt?
        outputs = kernel_matrix + self.bias
        return outputs

def get_kernel(kernel_type,input_size ,num_classes, gamma,r, degree):
    if kernel_type == 'linear':
        return LinearSVM(input_size, num_classes)
    elif kernel_type == 'rbf':
        return RBFSVM(input_size, num_classes, gamma)
    elif kernel_type == 'poly':
        return PolySVM(input_size, num_classes, gamma, r, degree)
    elif kernel_type == 'sigmoid':
        return SigmoidSVM(input_size, num_classes, gamma, r)
    elif kernel_type == 'custom':
        return CustomSVM(input_size, num_classes, gamma, r, degree)
    else:
        raise ValueError('không hỗ trợ kernel này')