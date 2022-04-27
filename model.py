import torch
import torch.nn as nn

class MultiPredictNet(nn.Module):
    def __init__(self, init_weight=False):
        super(MultiPredictNet, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 196, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(196),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.global_max_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.age_fc_layers = nn.Sequential(
            nn.Linear(in_features=196, out_features=25),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=1),
            nn.Sigmoid()
        )
        
        self.gender_fc_layers = nn.Sequential(
            nn.Linear(in_features=196, out_features=25, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=2, bias=True),
        )
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.global_max_pooling(x)
        x = torch.flatten(x, 1)
        m_age = self.age_fc_layers(x)
        m_gender = self.gender_fc_layers(x)
        return m_age, m_gender
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    model = MultiPredictNet(init_weight=True)
    print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

# model = nn.Sequential(
#     nn.Conv2d(1,20,5),
#     nn.ReLU(),
#     nn.Conv2d(20, 64, 5),
#     nn.ReLU()
# )

# print(model)
