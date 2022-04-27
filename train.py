import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AgeGenderDataset
from model import MultiPredictNet

train_on_gpu = True
 
ds = AgeGenderDataset(".\\dataset\\UTKFace")
num_train_samples = ds.num_of_samples()
batch_size = 32
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
model = MultiPredictNet(init_weight=True)
if train_on_gpu:
    model.cuda()

# 训练模型的次数
num_epochs = 25
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
model.train()


# 损失函数
mse_loss = torch.nn.MSELoss()
cross_loss = torch.nn.CrossEntropyLoss()
index = 0

# print(next(dataloader))

for epoch in  range(num_epochs):
    train_loss = 0.0
    for i_batch, sample_batched in enumerate(dataloader):
        images_batch, age_batch, gender_batch = \
            sample_batched['image'], sample_batched['age'], sample_batched['gender']
        if train_on_gpu:
            images_batch, age_batch, gender_batch = images_batch.cuda(), age_batch.cuda(), gender_batch.cuda()
        optimizer.zero_grad()
 
        images_batch = images_batch.to(torch.float32)
        # forward pass: compute predicted outputs by passing inputs to the model
        m_age_out_, m_gender_out_ = model(images_batch)
        age_batch = age_batch.view(-1, 1).to(torch.float)
        gender_batch = gender_batch.long()
 
 
        # calculate the batch loss
        loss = mse_loss(m_age_out_, age_batch) + cross_loss(m_gender_out_, gender_batch)
 
 
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
 
 
        # perform a single optimization step (parameter update)
        optimizer.step()
 
 
        # update training loss
        train_loss += loss.item()
        if index % 100 == 0:
            print('step: {} \tTraining Loss: {:.6f} '.format(index, loss.item()))
        index += 1
        
 
        # 计算平均损失
    train_loss = train_loss / num_train_samples
 
 
    # 显示训练集与验证集的损失函数
    print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))
 
 
# save model
model.eval()
torch.save(model, 'age_gender_model.pth')