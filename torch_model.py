import numpy as np
import cv2 as cv

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Net(nn.Module):

    def __init__(self, class_num, channel_num=1, batch_size=32):
        super(Net, self).__init__()
        self._batch_size = batch_size
        filter_11, filter_12, filter_2, filter_3, filter_4 = 32, 64, 128, 256, 512
        
        self.pool = nn.MaxPool2d(2, 2)
        self._dropout_03 = nn.Dropout(0.3)
        self._dropout_05 = nn.Dropout(0.5)

        self._conv_11 = nn.Conv2d(channel_num, filter_11, 7)
        self._conv_12 = nn.Conv2d(filter_11, filter_12, 5)
        self._norm1 = nn.BatchNorm2d(filter_12)

        self._conv2 = nn.Conv2d(filter_12, filter_2, 3)
        self._norm2 = nn.BatchNorm2d(filter_2)        
        

        self._conv3 = nn.Conv2d(filter_2, filter_3, 3)
        self._norm3 = nn.BatchNorm2d(filter_3)        
        
        self._conv2 = nn.Conv2d(filter_3, filter_4, 3)
        self._norm2 = nn.BatchNorm2d(filter_4)

        self._hidden1 = nn.Linear(filter_4*5*5, 1024)
        self._hidden2 = nn.Linear(1024, 2048)
        self._hidden3 = nn.Linear(2048, class_num)

    #TODO: deal with pic_size after convolution
    def forward(self, x):
        #print('before convolution ' + str(x.shape))
        x = F.relu(self._conv11(x))
        x = F.relu(self._conv12(x))
        x = self._norm1(x)
        x = self.pool(x)

        x = F.relu(self._conv2(x))
        x = self._norm2(x)
        x = self.pool(x)

        x = self._dropout_03(x)

        x = F.relu(self._conv3(x))
        x = self._norm3(x)
        x = self.pool(x)

        x = F.relu(self._conv4(x))
        x = self._norm4(x)
        x = self.pool(x)
        
        print('after convolution ' + str(x.shape))

        x = x.reshape(self._batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self._dropout_03(x)
        x = F.relu(self.fc2(x))
        x = self._dropout_05(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Emotion_Net:
    def __init__(self, n_classes, im_channels, batch_size=32):
        self._model = Net(n_classes, im_channels, batch_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, trainloader, testloader, n_epochs=50, 
                optimizer='adam', save_best=False, save_best_path=''):
        criterion = nn.CrossEntropyLoss()
        opt = optim.SGD(self._model.parameters(), lr=0.001)
        if optimizer=='adam' or optim=="Adam":
            opt = optim.Adam(self._model.parameters(), lr=0.005)
        best_accuracy = 0.0
        for epoch in range(n_epochs): 
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                # data is already transposed
                #inputs = inputs.transpose(1, 3)
                opt.zero_grad()
                
                outputs = self._model(inputs.float()) 
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()
                
                #loss_trace.append(loss.item())
                running_loss += loss.item()
                if i % 100 == 99:  
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data[0].to(self.device), data[1].to(self.device)
                            images = images.transpose(1, 3)
                            outputs = self._model(images.float())
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                        accuracy = correct / total
                        print('Accuracy of the network on the 10000 test images: %d %%' % (
                                100 * accuracy))
                        if save_best and best_accuracy < accuracy:
                            best_accuracy = accuracy
                            torch.save(self._model, save_best_path)

    def save_model(self, path):
        torch.save(self._model, path)

    def load_model(self, path):
        self._model = torch.load(path)
                    
