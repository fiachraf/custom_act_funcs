import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#if input negative, output = input, else output = 0 (i.e. for positives)
def neg_relu(x, inplace=False):
    #if inplace True then y is an alias, else y is a copy
    if inplace:
        y = x
    else:
        y = torch.clone(x)
    y[y>0] = 0
    return y

#has to be implemented as class
class NegReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()  #init the base class
        self.inplace = inplace
    def forward(self, input):
        return neg_relu(input, self.inplace)

#passed test for gradient propagation

#relu with a different slope
def dslope_relu(x, inplace=False, dslope=1.5):
    #could possibly make dslope another input variable and also potentially a trainable parameter
    #value of dslope should be changed and tested for different values
    # dslope = 1.5
    if inplace:
        y = x
    else:
        y = torch.clone(x)
    y[y<0] = 0
    return dslope * y
class DSlopeReLU(nn.Module):
    def __init__(self, inplace=False, dslope=1.5):
        super().__init__()
        self.inplace = inplace
        self.dslope = dslope
    def forward(self, input):
        return dslope_relu(input, self.inplace, self.dslope)

        #passed test for gradient propagation


#relu except the y intercept is changed and all positive values are increased/decreased by the value of the y intercept thus keeping the slope at 1
def diff_y_relu(x, inplace=False, y_int=0.5):
    #could possibly make y_int another input variable and also potentially a trainable parameter
    # y_int = 0.5
    if inplace:
        y = x
    else:
        y = torch.clone(x)
    y[y<0] = 0
    y[y>0] += y_int
    return y
class Diff_Y_ReLU(nn.Module):
    def __init__(self, inplace=False, y_int=0.5):
        super().__init__()
        self.inplace = inplace
        self.y_int = y_int
    def forward(self, input):
        return diff_y_relu(input, self.inplace, self.y_int)

        #passed test for gradient propagation

#output of function is like a hill, with the peak not necessarily in the middle, any inputs less than 0 or greater than 1 are set to 0
def pos_hill(x, inplace=False, turn_point=0.75):
    #could possibly make the slopes another input variable and also potentially a trainable parameter
    # need to change both the up_slope and down_slope at the same time to maintain the continuity of the line
    up_slope = 1/turn_point
    down_slope = -1/(1 - turn_point)
    if inplace:
        #modify in place then y is an alias
        y = x
    else:
        #don't modify in palce then y is a copy of x
        y = torch.clone(x)
    y[y<0] = 0
    y[y>1] = 0
    # the down_slope is also equal to the negative of the y intercept for the downward part of the hill
    y = torch.where(y<0.75, y*up_slope, y*down_slope - down_slope)
    return y
class Pos_Hill(nn.Module):
    def __init__(self, inplace=False, turn_point=0.75):
        super().__init__()
        self.inplace = inplace
        self.turn_point = turn_point
    def forward(self, input):
        return pos_hill(input, self.inplace, self.turn_point)

    #passed test for gradient propagation


#positive inputs are treated like relu, negative inputs are multiplied by a decimal
def small_neg(x, inplace=False, neg_slope=0.2):
    #neg_slope could be varied and also be used as a trainable parameter
    # neg_slope = 0.2
    if inplace:
        y = x
    else:
        y = torch.clone(x)
    y[y<0] *= neg_slope
    return y
class Small_Neg(nn.Module):
    def __init__(self, inplace=False, neg_slope=0.75):
        super().__init__()
        self.inplace = inplace
        self.neg_slope = neg_slope
    def forward(self, input):
        return small_neg(input, self.inplace, self.neg_slope)

        #passed test for gradient propagation




if __name__ == '__main__':

    #test gradient calculation
    x=torch.tensor([-1.0, 0.5], requires_grad=True)
    y = torch.tensor([0.75, 0.1], requires_grad=True)
    # x=torch.tensor([-1.0], requires_grad=True)
    # y = torch.tensor([0.75], requires_grad=True)
    # z = x * y
    z = x - y
    # a = neg_relu(z)
    # a = dslope_relu(z)
    # a = diff_y_relu(z)
    # a = pos_hill(z)
    # a = small_neg(z, False, 0.2)
    a = neg_relu(z)

    # grad_1 = torch.autograd.grad(a, (x, y, z, a), create_graph=True)
    # print(f"x: {x.item()}, y: {y.item()}, z: {z.item()}, a: {a.item()}")
    print(f"x: {x}, y: {y}, z: {z}, a: {a}")
    # print(f"grad_1: {grad_1}")