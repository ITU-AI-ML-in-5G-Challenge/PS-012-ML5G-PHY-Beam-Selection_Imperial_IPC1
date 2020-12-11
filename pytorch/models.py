from torch import nn


class Lidar2D(nn.Module):
    def __init__(self):
        super(Lidar2D, self).__init__()
        self.channels = 5
        self.conv1 = nn.Conv2d(1, self.channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu1 = nn.PReLU(num_parameters=self.channels)
        self.conv2 = nn.Conv2d(self.channels, self.channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.channels)
        self.relu2 = nn.PReLU(num_parameters=self.channels)
        self.conv3 = nn.Conv2d(self.channels, self.channels, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(self.channels)
        self.relu3 = nn.PReLU(num_parameters=self.channels)
        self.conv4 = nn.Conv2d(self.channels, self.channels, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(self.channels)
        self.relu4 = nn.PReLU(num_parameters=self.channels)
        self.conv5 = nn.Conv2d(self.channels, self.channels, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(self.channels)
        self.relu5 = nn.PReLU(num_parameters=self.channels)
        self.conv6 = nn.Conv2d(self.channels, self.channels, 3, (1, 2), 1)
        self.bn6 = nn.BatchNorm2d(self.channels)
        self.relu6 = nn.PReLU(num_parameters=self.channels)
        self.linear7 = nn.Linear(125 * self.channels, 16)
        self.relu7 = nn.ReLU()
        self.linear8 = nn.Linear(16, 256)


    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        #
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = x.view(-1, 125 * self.channels)
        x = self.linear7(x)
        x = self.relu7(x)
        x = self.linear8(x)

        return x
