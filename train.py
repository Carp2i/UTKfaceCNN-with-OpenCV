import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from dataset import utkface

from tqdm import tqdm

from model import MultiPredictNet


def main():

    torch.backends.cudnn.enable =True
    torch.backends.cudnn.benchmark = True

    # train_on_gpu = True
    batch_size = 16


    # runtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # if running in win os plz set the num_workers to 0
    utkdataset = utkface("./dataset/UTKFace")

    val_num = int(len(utkdataset)*0.2)
    train_num = len(utkdataset) - val_num

    # Split the data in train and val
    train_data, val_data = random_split(utkdataset, [train_num, val_num])



    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # validate set
    # valset = AgeGenderDataset("./dataset/UTKFace")
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0) 

    # train_num = trainset.num_of_samples()
    # val_num = valset.num_of_samples()

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    ## 检查验证集的代码
    # 这里可以用 opencv 的方法查看样本
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    # def imshow(img):
    #     img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()

    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = MultiPredictNet(init_weight=True)
    net.to(device)
    # 训练模型的次数
    # epoch_num = 25
    epoch_num = 3
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    # 损失函数
    mse_loss = torch.nn.MSELoss()
    cross_loss = torch.nn.CrossEntropyLoss()
    # index = 0

    for epoch in range(epoch_num):
        train_loss = 0.0
        net.train()
        train_bar = tqdm(train_loader, file=sys.stdout)
        for i, data in enumerate(train_bar):
            images_batch, age_batch, gender_batch = \
                data['image'], data['age'], data['gender']
            
            inputs = images_batch.to(device=device)
            age_label =  age_batch.to(device=device)
            gender_label = gender_batch.to(device=device)
            
            # print(images_batch.size(), age_batch.size(), gender_batch.size())
            optimizer.zero_grad()
            inputs = inputs.to(torch.float)

            # forward pass: compute predicted outputs by passing inputs to the model
            m_age_out_, m_gender_out_ = net(inputs)
            
            age_label = age_label.view(-1, 1).to(torch.float)
            gender_label = gender_label.long()
            
            # calculate the batch loss
            # print(m_age_out_.size(), age_batch.size(), m_gender_out_.size(), gender_batch.size())
            loss = mse_loss(m_age_out_, age_label) + cross_loss(m_gender_out_, gender_label)
    
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()

            # print statistics
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epoch_num, loss)

            # if index % 100 == 0:
            #     print('step: {} \tTraining Loss: {:.6f} '.format(index, loss.item()))
            # index += 1
            

        # validate part
        net.eval()
        acc_gender = 0.0       # accumulate accurate sum/epoch
        loss_age = 0.0
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for j, val_set in enumerate(val_bar):
                val_images, val_age, val_gender = \
                    val_set['image'], val_set['age'], val_set['gender']
                
                val_images = val_images.to(device=device)
                val_age = val_age.to(device=device)
                val_gender = val_gender.to(device=device)

                val_inputs = val_images.to(torch.float)
                # val_gender = val_gender.to()
                val_age_out, val_gender_out = net(val_inputs)

                val_age = val_age.view(-1, 1).to(torch.float)
                val_gender = val_gender.long()

                pre_gender = torch.max(val_gender_out, dim=1)[1]
                acc_gender += torch.eq(pre_gender, val_gender).sum().item()

                # print(pre_gender, val_gender)

                # val_loss
                val_loss += mse_loss(val_age_out, val_age) + cross_loss(val_gender_out, val_gender.to(device))
                loss_age += mse_loss(val_age_out, val_age)

        # acc_gender /= val_num

        print("[epoch %d] train_loss: %.3f  val_loss: %.3f" %
                (epoch + 1, train_loss / i, val_loss / j))
        print("[epoch %d] gender_acc: %.3f  mse_age: %.3f" %
                (epoch + 1, acc_gender/val_num, loss_age/val_num))
    
    
    # save model
    torch.save(net.state_dict(), 'age_gender_model.pth')

if __name__ == "__main__":
    main()