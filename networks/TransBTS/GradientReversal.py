import torch
from torch.autograd import Function
import time 

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        out = grad_output.neg()
        return out 


def grad_reverse(x):
    return GradReverse.apply(x)
