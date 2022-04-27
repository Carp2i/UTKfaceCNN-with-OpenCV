import torch
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
onnx_path = 'onnx_model.onnx'
# net = MultiPredictNet().to(device)
weights_path = "./age_gender_model.pth"
net = torch.load(weights_path)
net.eval()
d = torch.rand(1, 3, 64, 64).to(device)
print(net)

torch.onnx.export(net, d, onnx_path)
