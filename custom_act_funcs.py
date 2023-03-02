import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#I had initially include an argument for each function to have it operate in place but it didn't work properly for some so I just removed that
#if input negative, output = input, else output = 0 (i.e. for positives)
def neg_relu(x):
    # y = torch.clone(x)
    #this index operation counts as an inplace operation and so it doesn't propagate gradients correctly without the previous clone step and even then it might not be propagaing gradients properly
    # y[y>0] = 0
    #however the torch.where function propagates gradients correctly and also removes the extra memory I'm probably using up from the clone statement
    x = torch.where(x<0, x, 0)
    return x

#has to be implemented as class in order to be used in a NeuralNetwork
class NegReLU(nn.Module):
    def __init__(self):
        super().__init__()  #init the base class

    def forward(self, input):
        return neg_relu(input)

#passed test for gradient propagation

#relu with a different slope
def dslope_relu(x, dslope=1.5):
    #could possibly make dslope another input variable and also potentially a trainable parameter
    #value of dslope should be changed and tested for different values
    x = torch.where(x < 0, 0, x*dslope)
    return x
class DSlopeReLU(nn.Module):
    def __init__(self, dslope=1.5):
        super().__init__()
        self.dslope = dslope
    def forward(self, input):
        return dslope_relu(input, self.dslope)

        #passed test for gradient propagation


#relu except the y intercept is changed and all positive values are increased/decreased by the value of the y intercept thus keeping the slope at 1
def diff_y_relu(x, y_int=0.5):
    #could possibly make y_int another input variable and also potentially a trainable parameter
    x = torch.where(x<0, 0, x + y_int)
    return x
class Diff_Y_ReLU(nn.Module):
    def __init__(self, y_int=0.5):
        super().__init__()
        self.y_int = y_int
    def forward(self, input):
        return diff_y_relu(input, self.y_int)

        #passed test for gradient propagation

#output of function is like a hill, with the peak not necessarily in the middle, any inputs less than 0 or greater than 1 are set to 0
def pos_hill(x, turn_point=0.75):
    #could possibly make the slopes another input variable and also potentially a trainable parameter
    # need to change both the up_slope and down_slope at the same time to maintain the continuity of the line
    up_slope = 1/turn_point
    down_slope = -1/(1 - turn_point)
    # set values below 0 or above 1 to 0
    x = torch.where((x < 0) | (x > 1), 0, x)
    # the down_slope is also equal to the negative of the y intercept for the downward part of the hill
    x = torch.where(x<0.75, x*up_slope, x*down_slope - down_slope)
    return x
class Pos_Hill(nn.Module):
    def __init__(self, turn_point=0.75):
        super().__init__()
        self.turn_point = turn_point
    def forward(self, input):
        return pos_hill(input, self.turn_point)

    #passed test for gradient propagation


#positive inputs are treated like relu, negative inputs are multiplied by a decimal
def small_neg(x, neg_slope=0.2):
    #neg_slope could be varied and also be used as a trainable parameter
    x = torch.where(x > 0, x, x * neg_slope)
    return x
class Small_Neg(nn.Module):
    def __init__(self, neg_slope=0.75):
        super().__init__()
        self.neg_slope = neg_slope
    def forward(self, input):
        return small_neg(input, self.neg_slope)

        #passed test for gradient propagation

#pos hill except with more control over wher the hill starts, peaks and ends. Each of these points is specified using tuples(x,y)
#start and end point can have different y values
def pos_hill_v2(x, start_p=(0.0,0.0), peak_p=(1.0,1.0), end_p=(2.0,0.0)):

    #changing the tuples to be tensors with the same dtype as the input tensor x, as torch.where requires all arguments to have the same dtype
    t_start_p = x.new([start_p[0], start_p[1]])
    t_peak_p = x.new([peak_p[0], peak_p[1]])
    t_end_p = x.new([end_p[0], end_p[1]])

    #determining the slope and y intercepts for the lines from the given points
    up_slope = (t_peak_p[1] - t_start_p[1]) / (t_peak_p[0] - t_start_p[0])
    down_slope = (t_end_p[1] - t_peak_p[1]) / (t_end_p[0] - t_peak_p[0])
    c1 = t_start_p[1] - up_slope * t_start_p[0]
    c2 = t_end_p[1] - down_slope * t_end_p[0]

    #creating other tensors, one for multiplication and one for addition.
    #By doing it with tensors it makes it much easier to do Backpropagation
    #the multiplication is the multiplication by the slope and the addition is the adding the y intercept
    #could change to make the turning point multiplied by 0

    mul_tens = torch.empty_like(x)
    add_tens = torch.empty_like(x)

    #creating a tensor with the same dtype and device as input x, to use to get the 0 slope, regardless of the input points used
    t_zero = x.new([0.0])

    #for inputs less than the start point (by default 0)
    mul_tens = torch.where(x<t_start_p[0], t_zero[0], mul_tens)
    #for inputs between the start and mid points
    mul_tens = torch.where((x>=t_start_p[0]) & (x<t_peak_p[0]), up_slope, mul_tens)
    #for inputs between the mid and end points
    mul_tens = torch.where((x>=t_peak_p[0]) & (x<t_end_p[0]), down_slope, mul_tens)
    #for inputs greater than the end point (by default 2.0)
    mul_tens = torch.where(x>=t_end_p[0], t_zero[0], mul_tens)

    #doing the same thing for the addition tensor
    add_tens = torch.where(x<t_start_p[0], t_start_p[1], add_tens)
    add_tens = torch.where((x>=t_start_p[0]) & (x<t_peak_p[0]), c1, add_tens)
    add_tens = torch.where((x>=t_peak_p[0]) & (x<t_end_p[0]), c2, add_tens)
    add_tens = torch.where(x>=t_end_p[0], t_end_p[1], add_tens)

    x = torch.mul(x, mul_tens)
    x = torch.add(x, add_tens)
    return x
class Pos_Hill_V2(nn.Module):
    def __init__(self, start_p=(0.0,0.0), peak_p=(1.0,1.0), end_p=(2.0,0.0)):
        super().__init__()
        self.start_p = start_p
        self.peak_p = peak_p
        self.end_p = end_p
    def forward(self, input):
        return pos_hill_v2(input, self.start_p, self.peak_p, self.end_p)

#pos hill v2 except the slope at turning points including the peak are set to 0.
def pos_hill_v3(x, start_p=(0.0,0.0), peak_p=(1.0,1.0), end_p=(2.0,0.0)):

    #changing the tuples to be tensors with the same dtype as the input tensor x, as torch.where requires all arguments to have the same dtype
    t_start_p = x.new([start_p[0], start_p[1]])
    t_peak_p = x.new([peak_p[0], peak_p[1]])
    t_end_p = x.new([end_p[0], end_p[1]])

    #determining the slope and y intercepts for the lines from the given points
    up_slope = (t_peak_p[1] - t_start_p[1]) / (t_peak_p[0] - t_start_p[0])
    down_slope = (t_end_p[1] - t_peak_p[1]) / (t_end_p[0] - t_peak_p[0])
    c1 = t_start_p[1] - up_slope * t_start_p[0]
    c2 = t_end_p[1] - down_slope * t_end_p[0]

    #creating other tensors, one for multiplication and one for addition.
    #By doing it with tensors it makes it much easier to do Backpropagation
    #the multiplication is the multiplication by the slope and the addition is the adding the y intercept
    #could change to make the turning point multiplied by 0

    mul_tens = torch.empty_like(x)
    add_tens = torch.empty_like(x)

    #creating a tensor with the same dtype and device as input x, to use to get the 0 slope, regardless of the input points used
    t_zero = x.new([0.0])

    #for inputs less than the start point (by default 0)
    mul_tens = torch.where(x<=t_start_p[0], t_zero[0], mul_tens)
    #for inputs between the start and mid points
    mul_tens = torch.where((x>t_start_p[0]) & (x<t_peak_p[0]), up_slope, mul_tens)
    #getting the slope at the peak turning point to be 0
    mul_tens = torch.where(x==t_peak_p[0], t_zero, mul_tens)
    #for inputs between the mid and end points
    mul_tens = torch.where((x>t_peak_p[0]) & (x<t_end_p[0]), down_slope, mul_tens)
    #for inputs greater than the end point (by default 2.0)
    mul_tens = torch.where(x>=t_end_p[0], t_zero[0], mul_tens)

    #doing the same thing for the addition tensor
    add_tens = torch.where(x<=t_start_p[0], t_start_p[1], add_tens)
    add_tens = torch.where((x>t_start_p[0]) & (x<t_peak_p[0]), c1, add_tens)
    add_tens = torch.where(x==t_peak_p[0], t_peak_p[1], add_tens)
    add_tens = torch.where((x>t_peak_p[0]) & (x<t_end_p[0]), c2, add_tens)
    add_tens = torch.where(x>=t_end_p[0], t_end_p[1], add_tens)

    x = torch.mul(x, mul_tens)
    x = torch.add(x, add_tens)
    return x
class Pos_Hill_V3(nn.Module):
    def __init__(self, start_p=(0.0,0.0), peak_p=(1.0,1.0), end_p=(2.0,0.0)):
        super().__init__()
        self.start_p = start_p
        self.peak_p = peak_p
        self.end_p = end_p
    def forward(self, input):
        return pos_hill_v3(input, self.start_p, self.peak_p, self.end_p)



def double_hill(x, start_p=(-2,0), peak_1=(-1,1), mid_p=(0,0), peak_2=(1,1), end_p=(2,0)):
        y = torch.clone(x)
        up_slope_1 = (peak_1[1] - start_p[1]) / (peak_1[0] - start_p[0])
        down_slope_1 = (mid_p[1] - peak_1[1]) / (mid_p[0] - peak_1[0])
        c1 = start_p[1] - up_slope_1 * start_p[0]
        c2 = mid_p[1] - down_slope_1 * mid_p[0]

        up_slope_2 = (peak_2[1] - mid_p[1]) / (peak_2[0] - mid_p[0])
        down_slope_2 = (end_p[1] - peak_2[1]) / (end_p[0] - peak_2[0])
        c3 = mid_p[1] - up_slope_2 * mid_p[0]
        c4 = end_p[1] - down_slope_2 * end_p[0]


        left_1 = torch.where((y<=peak_1[0]) & (y>start_p[0]), y* up_slope_1 + c1, start_p[1])
        right_1 = torch.where((y>peak_1[0]) & (y<mid_p[0]), y* down_slope_1 + c2, mid_p[1])
        # truth_table = torch.where(torch.eq(left, y))
        # rejoin = torch.where((a := 1 if truth_table else 0) > 0, left, right)
        new_1 = torch.where(left_1>right_1, left_1, right_1)
        # y[y>end_p[1]] = end_p[1]
        left_2 = torch.where((y<=peak_2[0]) & (y>mid_p[0]), y* up_slope_2 + c3, mid_p[1])
        right_2 = torch.where((y>peak_2[0]) & (y<end_p[0]), y* down_slope_2 + c4, end_p[1])
        new_2 = torch.where(left_2>right_2, left_2, right_2)

        new_3 = torch.where(new_2>new_1, new_2, new_1)
        return new_3
class Double_Hill(nn.Module):
    def __init__(self, start_p=(-2,0), peak_1=(-1,1), mid_p=(0,0), peak_2=(1,1), end_p=(2,0)):
        super().__init__()
        self.start_p = start_p
        self.peak_1 = peak_1
        self.mid_p = mid_p
        self.peak_2 = peak_2
        self.end_p = end_p
    def forward(self, input):
        return double_hill(input, self.start_p, self.peak_1, self.mid_p, self.peak_2, self.end_p)

def val_hill(x, start_p=(-2,0), val_p=(-1,-1), peak_p=(1,1), end_p=(2,0)):
    y = torch.clone(x)
    down_slope_1 = (val_p[1] - start_p[1]) / (val_p[0] - start_p[0])
    up_slope_1 = (peak_p[1] - val_p[1]) / (peak_p[0] - val_p[0])
    c1 = start_p[1] - down_slope_1 * start_p[0]
    c2 = val_p[1] - up_slope_1 * val_p[0]

    # up_slope_2 = (peak_2[1] - mid_p[1]) / (peak_2[0] - mid_p[0])
    down_slope_2 = (end_p[1] - peak_p[1]) / (end_p[0] - peak_p[0])
    c3 = peak_p[1] - down_slope_2 * peak_p[0]
    # c4 = end_p[1] - down_slope_2 * end_p[0]

    #does it in 2 main sections the valley and hill sections. For the hill sections it sets any other value that isn't on the hill to be at the same value as the bottom of the valley so that in the last line for new_3, the inequality holds as the valley values will be greater than the hill values.
    #This function is done with four subsections instead of three overall sectins as a "middle" section crosses the y axis which makes it difficult to then combine the sections using the torch.where inequalities

    mid_p = ((peak_p[0] + val_p[0]) / 2, (peak_p[1] + val_p[1]) / 2)
    left = torch.where((y<=val_p[0]) & (y>start_p[0]), y* down_slope_1 + c1, start_p[1])
    mid_left = torch.where((y>val_p[0]) & (y<mid_p[0]), y* up_slope_1 + c2, start_p[1])
    # truth_table = torch.where(torch.eq(left, y))
    # rejoin = torch.where((a := 1 if truth_table else 0) > 0, left, right)
    new_1 = torch.where(left<mid_left, left, mid_left)
    # y[y>end_p[1]] = end_p[1]
    right = torch.where((y>peak_p[0]) & (y<end_p[0]), y* down_slope_2 + c3, val_p[1])
    mid_right = torch.where((y>mid_p[0]) & (y<=peak_p[0]), y* up_slope_1 + c2, val_p[1])
    new_2 = torch.where(mid_right>right, mid_right, right)

    new_3 = torch.where(new_1>new_2, new_1, new_2)
    return new_3
class Val_Hill(nn.Module):
    def __init__(self, start_p=(-2,0), val_p=(-1,-1), peak_p=(1,1), end_p=(2,0)):
        super().__init__()
        self.start_p = start_p
        self.val_p = val_p
        self.peak_p = peak_p
        self.end_p = end_p
    def forward(self, input):
        return val_hill(input, self.start_p, self.val_p, self.peak_p, self.end_p)

act_func_dict = {"neg_relu" : neg_relu, "dslope_relu" : dslope_relu, "diff_y_relu" : diff_y_relu, "pos_hill" : pos_hill, "small_neg" : small_neg, "pos_hill_v2" : pos_hill_v2, "pos_hill_v3" : pos_hill_v3, "double_hill": double_hill, "val_hill" : val_hill}
act_class_dict = {"NegReLU": NegReLU(), "DSlopeReLU": DSlopeReLU(), "Diff_Y_ReLU": Diff_Y_ReLU(), "Pos_Hill" : Pos_Hill(), "Small_Neg": Small_Neg()}


if __name__ == '__main__':

    #test gradient calculation
    x=torch.tensor([[-1.0, -1.1, -1.2], [0.5, 0.6, 0.7]], requires_grad=True)
    y = torch.tensor([[0.75, 0.76, 0.77], [0.1, 0.2, 0.3]], requires_grad=True)
    y1 = torch.tensor([[0.85, 0.86, 0.87], [0.91, 0.92, 0.93]], requires_grad=True)
    y7 = torch.tensor([[0.85, 0.86, 0.87], [0.91, 0.92, 0.93]])
    print(f"y7: {y7}")
    y7_v = y7.view(-1)
    print(f"y7_v: {y7_v}")
    y7_v[0] = 0
    print(f"y7_v: {y7_v}")
    # x=torch.tensor([-1.0, -1.1, -1.2], requires_grad=True)
    # y = torch.tensor([0.75, 0.76, 0.77], requires_grad=True)
    # y1 = torch.tensor([0.85, 0.86, 0.87], requires_grad=True)
    x_v = x.view(-1)
    y_v = y.view(-1)
    y1_v = y1.view(-1)
    new_1 = torch.empty_like(x_v)
    for index, val in enumerate(x_v):
        print(index, val)
        new_1[index] = val * index
    print(f"new_1: {new_1}")
    # print(f"new_1[0]: {new_1[0]}, new_1.backward():{new_1.backward()}, new_1.grad_fn: {new_1.grad_fn}, new_1.grad_fn.next_functions: {new_1.grad_fn.next_functions}, ")
    print(f"new_1[0]: {new_1[0]}, new_1.grad_fn: {new_1.grad_fn}, new_1.grad_fn.next_functions: {new_1.grad_fn.next_functions}, ")

    print(f"x: {x}")
    print(f"x_v: {x_v}")
    # x_v[-1] = x_v[-1] + 7
    print(f"x: {x}")
    print(f"x_v: {x_v}")
    # x=torch.tensor([-1.0], requires_grad=True)
    # y = torch.tensor([0.75], requires_grad=True)
    # z = x * y
    z = x - y
    # a = neg_relu(z)
    # a = dslope_relu(z)
    # a = diff_y_relu(z)
    # a = pos_hill(z)
    # a = small_neg(z, False, 0.2)
    a = pos_hill_v2(z)
    # a = pos_hill_v2(z)
    stacked = torch.stack((x,y,y1), -1)
    print(f"stacked.size(): {stacked.size()}")
    print(f"stacked: {stacked}")
    for i, p in enumerate(stacked):
        print(p)
    stacked_v = torch.stack((x_v,y_v,y1_v), -1)
    print(f"stacked_v.size(): {stacked_v.size()}")
    print(f"stacked_v: {stacked_v}")
    for i, p in enumerate(stacked_v):
        print(p)

    # grad_1 = torch.autograd.grad(a, (x, y, z, a), create_graph=True)
    # print(f"x: {x.item()}, y: {y.item()}, z: {z.item()}, a: {a.item()}")
    print(f"x: {x}, y: {y}, z: {z}, a: {a}")
    # print(f"a[0]: {a[0]}, a.backward():{a.backward()}, a.grad_fn: {a.grad_fn}, a.grad_fn.next_functions: {a.grad_fn.next_functions}, ")
    # print(f"grad_1: {grad_1}")
    from torch.autograd import gradcheck

    # x2 = torch.rand(256, dtype=torch.double, requires_grad=True)
    # x3 = torch.rand(256, dtype=torch.double, requires_grad=True)
    # x4 = torch.rand(256, dtype=torch.double, requires_grad=True)
    # x5 = torch.rand(256, dtype=torch.double, requires_grad=True)

    #gradcheck first arg is the function to check, 2nd arg is a tuple of inputs to use for testing, typically a torch.rand() tensor, if the test function only takes one input argument then the tuple should be in the form (torch.rand(),)
    # grad_test = gradcheck(pos_hill_v2, (x2,x3,x4,x5))
    # print(f"grad_test: {grad_test}")

    make_plots = input("Do you want to make plots of the activation functions? Y/N: ")
    if make_plots == "Y":
        import matplotlib.pyplot as plt
        import numpy as np
        # plt.style.use('_mpl-gallery')
        for name, func in act_func_dict.items():
            #creating data to plot
            x_range = torch.arange(-5,5.1,0.005)
            y_vals = func(x_range)
            # print(name)
            #need [0] to get overall length of one dimensional vector as .size() returns a vector
            # for i in range(x_range.size()[0]):
            #     print(x_range[i], ",", y_vals[i])
            # print("\n")
            # print(y_vals)
            #plotting the data
            fig, ax = plt.subplots()
            plt.grid(True)
            ax.plot(x_range, y_vals, linewidth=2.0)
            ax.set(xlim=(-5, 5), xticks=np.arange(-5, 5.1), ylim=(-5, 5), yticks=np.arange(-5, 5.1))
            plt.title(name)
            plt.savefig(f"{name}_plot.png", format='png')
            #show has to come after savefig if you want to save the figure being shown
            plt.show()
