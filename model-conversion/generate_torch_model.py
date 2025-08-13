import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 假设分类 10 类

    def forward(self, x):
        x = F.relu(self.conv1(x))      # [B, 16, 28, 28]
        x = F.max_pool2d(x, 2)         # [B, 16, 14, 14]
        x = F.relu(self.conv2(x))      # [B, 32, 14, 14]
        x = F.max_pool2d(x, 2)         # [B, 32, 7, 7]
        x = x.view(x.size(0), -1)      # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # 创建模型
    model = SimpleCNN()

    # 随机初始化参数（PyTorch 默认已经是随机初始化，这里显式调用一下）
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # 保存为 pth 格式
    torch.save(model.state_dict(), "simple_cnn.pth")
    print("模型参数已保存到 simple_cnn.pth")
