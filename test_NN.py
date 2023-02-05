import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import time

#import custom activation functions
import custom_act_funcs

act_funcs_list = [custom_act_funcs.NegReLU()]
act_funcs_dict = {'ReLU' : nn.ReLU(),
                    'NegReLU' : custom_act_funcs.NegReLU(),
                    'Dslope_0.25' : custom_act_funcs.DSlopeReLU(False,0.25),
                    'Dslope_0.5' : custom_act_funcs.DSlopeReLU(False,0.5),
                    'Dslope_0.75' : custom_act_funcs.DSlopeReLU(False,0.75),
                    'Dslope_1.0' : custom_act_funcs.DSlopeReLU(False,1.0),
                    'Dslope_1.25' : custom_act_funcs.DSlopeReLU(False,1.25),
                    'Dslope_1.5' : custom_act_funcs.DSlopeReLU(False,1.5),
                    'DiffYRelu_0.25' : custom_act_funcs.Diff_Y_ReLU(False,0.25),
                    'DiffYRelu_0.5' : custom_act_funcs.Diff_Y_ReLU(False,0.5),
                    'DiffYRelu_0.75' : custom_act_funcs.Diff_Y_ReLU(False,0.75),
                    'DiffYRelu_1.0' : custom_act_funcs.Diff_Y_ReLU(False,1.0),
                    'PosHill_0.25' : custom_act_funcs.Pos_Hill(False,0.25),
                    'PosHill_0.5' : custom_act_funcs.Pos_Hill(False,0.5),
                    'PosHill_0.75' : custom_act_funcs.Pos_Hill(False,0.75),
                    'SmallNeg_0.1' : custom_act_funcs.Small_Neg(False,0.1),
                    'SmallNeg_0.2' : custom_act_funcs.Small_Neg(False,0.2),
                    'SmallNeg_0.3' : custom_act_funcs.Small_Neg(False,0.3),
                    'SmallNeg_0.4' : custom_act_funcs.Small_Neg(False,0.4),
                    'SmallNeg_0.5' : custom_act_funcs.Small_Neg(False,0.5)
    }

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

#there are 60,000 images in the training dataset

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#Creating the model
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



#creating training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    train_accuracy_list.append(correct)
    train_loss_list.append(train_loss)


#creating testing loop
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    test_accuracy_list.append(correct)
    test_loss_list.append(test_loss)

import csv

with open("ResNet_test_results.csv", mode="w") as csv_file:
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(["Activation Function", "Epoch", "Training Accuracy", "Test Accuracy", "Training Loss", "Test Loss", "Time(s)"])



    # for each activation function
    for text, func in act_funcs_dict.items():

    #test each function 5 times in order to caluclate statistics
        for i in range(1,6):
        # Define model
        #small model
            # class NeuralNetwork(nn.Module):
            #     def __init__(self):
            #         super().__init__()
            #         self.flatten = nn.Flatten()
            #         self.linear_relu_stack = nn.Sequential(
            #             nn.Linear(28*28, 512),
            #             #func is the activation that is going to be used
            #             func,
            #             nn.Linear(512, 512),
            #             func,
            #             nn.Linear(512, 10)
            #         )
            #
            #     def forward(self, x):
            #         x = self.flatten(x)
            #         logits = self.linear_relu_stack(x)
            #         return logits

            #larger resnet inspired model to test if my functions scale well to deep models like ReLU does
            #create the ResidualBlock for use in the ResNet
            class ResidualBlock(nn.Module):
                def __init__(self, in_channels, out_channels, stride=1, downsample=None):
                    super().__init__()

                    self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(out_channels),
                        #my chosen activation function
                        func
                    )
                    self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels)
                    )
                    self.downsample = downsample
                    self.act_func = func
                    self.out_channels = out_channels

                def forward(self, x):
                    residual = x
                    out = self.conv1(x)
                    out = self.conv2(out)
                    if self.downsample:
                        residual = self.downsample(x)
                    out += residual
                    out = self.act_func(out)
                    return out

            class ResNet(nn.Module):
                def __init__(self, block, layers, num_classes=10):
                    super().__init__()

                    self.inplanes  = 64
                    self.conv1 = nn.Sequential(
                        #FashionMNIST dataset images are greyscale so 1 input channel
                        nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3),
                        nn.BatchNorm2d(64),
                        func
                    )
                    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
                    self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
                    self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
                    self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
                    # self.avgpool = nn.AvgPool2d(7, stride=1)
                    #had to change to 3 cos the FashionMNIST images start off small and would be too small(less than 0 in size) if let at 7
                    self.avgpool = nn.AvgPool2d(3, stride=1)
                    self.fc = nn.Linear(512, num_classes)


                def _make_layer(self, block, planes, blocks, stride=1):
                    downsample = None
                    if stride != 1 or self.inplanes != planes:
                        downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                        nn.BatchNorm2d(planes)
                        )
                    layers = []
                    layers.append(block(self.inplanes, planes, stride, downsample))
                    self.inplanes = planes
                    for i in range(1, blocks):
                        layers.append(block(self.inplanes, planes))

                    return nn.Sequential(*layers)

                def forward(self, x):
                    x = self.conv1(x)
                    x = self.maxpool(x)
                    x = self.layer0(x)
                    x = self.layer1(x)
                    #need to uncomment these lines and previous lines if I want to have the intended 4 layers
                    # x = self.layer2(x)
                    # x = self.layer3(x)

                    x = self.avgpool(x)
                    x = x.view(x.size(0), -1)
                    x = self.fc(x)

                    return x




            # model = ResNet(ResidualBlock, [3,1,2,4]).to(device)
            model = ResNet(ResidualBlock, [3,4,6,3]).to(device)
            print(model)

            #Optimizing the model parameters
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

            train_accuracy_list =[]
            test_accuracy_list =[]
            train_loss_list =[]
            test_loss_list =[]


            #training and testing the network
            epochs = 5
            for t in range(epochs):
                loop_start_time = time.perf_counter()
                print(f"Epoch {t+1}\n-------------------------------")
                train(train_dataloader, model, loss_fn, optimizer)
                test(test_dataloader, model, loss_fn)
                #write results to csv [act_func, epoch, train acc, test acc, train loss, test loss]
                #last element in accuracy and lost list should be results for the epoch that it has just done
                loop_end_time = time.perf_counter()
                csv_file_writer.writerow([text, t+1, train_accuracy_list[-1], test_accuracy_list[-1], train_loss_list[-1], test_loss_list[-1], loop_end_time - loop_start_time])
            print("Done!")

            #Saving the model
            #save the model with act func in the name as well iteration
            torch.save(model.state_dict(), f"{text}_ResNetmodel_{i}.pth")
            print(f"Saved PyTorch Model State to {text}_ResNetmodel_{i}.pth")

            #creating plot
            import matplotlib.pyplot as plt

            epochs = range(1, len(test_accuracy_list) +1)

            plt.plot(epochs, train_accuracy_list, "bo", label="Train acc")
            # plt.plot(epochs, val_acc, "b", label="Validation acc")
            plt.title(f"{text}_ResNetTrain accuracy_{i}")
            plt.legend()
            plt.savefig(f"{text}_ResNetTrain accuracy_{i}.png")
            # plt.figure()

            plt.clf()

            plt.plot(epochs, test_accuracy_list, "bo", label="Test acc")
            # plt.plot(epochs, val_acc, "b", label="Validation acc")
            plt.title(f"{text}_ResNetTest accuracy_{i}")
            plt.legend()
            plt.savefig(f"{text}_ResNetTest accuracy_{i}.png")

            plt.clf()
        # plt.figure()

# plt.plot(epochs, loss, "bo", label="Training loss")
# plt.plot(epochs, val_loss, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.legend()
# plt.savefig("loss.png")
# plt.show()

"""
#Loading the model
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

#Making predictions with the model
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
"""
