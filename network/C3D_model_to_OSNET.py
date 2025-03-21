import torch
import torch.nn as nn

# ================== 1. Define C3D ==================
class C3D(nn.Module):
    def __init__(self, pretrained=False):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)  # Thêm BatchNorm
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(128)  # Thêm BatchNorm
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn3a = nn.BatchNorm3d(256)  # Thêm BatchNorm
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn3b = nn.BatchNorm3d(256)  # Thêm BatchNorm
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc6 = nn.Linear(256, 256)  # Tăng kích thước embedding
        self.dropout = nn.Dropout(p=0.5)
        
        self.__init_weight()
        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3a(self.conv3a(x)))
        x = self.pool3(torch.relu(self.bn3b(self.conv3b(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc6(x)
        return x

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias"
                        }

        p_dict = torch.load('c3d-pretrained.pth')
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = C3D()
    outputs = net.forward(inputs)
    print(outputs.size())
    
    
