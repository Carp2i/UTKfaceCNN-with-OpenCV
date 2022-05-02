from ctypes import sizeof
import os

import torch
import torch.nn as nn
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
# from PIL import Image
import cv2 as cv

from model import MultiPredictNet

def main():
    max_age = 110

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load image 
    img_path = "./dataset/UTKFace/70_0_0_20170117091026248.jpg.chip.jpg"
    assert os.path.exists(img_path), "file '{}' does not exist.".format(img_path)
    # img = Image.open(img_path)
    img = cv.imread(img_path)
   
    # img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    # plt.imshow(img)

    ## view the example img
    cv.imshow("image", img)

    cv.waitKey(0)
    cv.destoryAllWindows()
    img = cv.resize(img, (64, 64))
    img = (np.float32(img) /255.0 - 0.5) / 0.5
    # H, W, C to C, H, W
    img = img.transpose((2, 0, 1))
    # print(img.shape)

    # [N, C, H, W]
    # expand batch dimension
    # img = cv.resize(img, (64, 64))
    # img = (np.float32(img) /255.0 - 0.5) / 0.5
    # H, W, C to C, H, W
    # img = img.transpose((2, 0, 1))
    img_in = torch.from_numpy(img)
    img_in = torch.unsqueeze(img_in, dim=0)
    # print(img_in.shape)
    
    weight_path = "age_gender_model.pth"
    assert os.path.exists(weight_path), "file '{}' does not exist.".format(weight_path)
    net = MultiPredictNet().to(device)
    net.load_state_dict(torch.load(weight_path))

    # predict
    net.eval()
    with torch.no_grad():
        # predict class
        out_age_raw, out_gender_raw = net(img_in.to(device))[0].cpu(), net(img_in.to(device))[1].cpu() 
        # print(net(img_in.to(device)))
        out_age = torch.squeeze(out_age_raw).cpu() * max_age
        out_gender = torch.squeeze(out_gender_raw).cpu()
        pred_gender = torch.softmax(out_gender, dim=0)
        predict_gender = torch.argmax(pred_gender).numpy()
        print(int(out_age), predict_gender)

if __name__ == "__main__":
    main()
