import os
import numpy as np
import cv2 as cv


import torch
from torchvision import transforms
from torch.utils.data import Dataset


max_age = 110
class utkface(Dataset):
     def __init__(self, root_dir):
         self.transform = transforms.Compose([transforms.ToTensor()])
         img_files = os.listdir(root_dir)
         nums_ = len(img_files)
         # age: 0 ~116, 0 :male, 1 :female
         self.ages = []
         self.genders = []
         self.images = []
         index = 0
         for file_name in img_files:
             age_gender_group = file_name.split("_")
             age_ = age_gender_group[0]
             gender_ = age_gender_group[1]
             self.genders.append(np.float32(gender_))
             self.ages.append(np.float32(age_)/max_age)
             self.images.append(os.path.join(root_dir, file_name))
             index += 1
 
 
     def __len__(self):
         return len(self.images)
 
 
     def num_of_samples(self):
         return len(self.images)
 
 
     def __getitem__(self, idx):
         if torch.is_tensor(idx):
             idx = idx.tolist()
             image_path = self.images[idx]
         else:
             image_path = self.images[idx]
         img = cv.imread(image_path)  # BGR order
         h, w, c = img.shape
         # rescale
         img = cv.resize(img, (64, 64))
         img = (np.float32(img) /255.0 - 0.5) / 0.5
         # H, W, C to C, H, W
         img = img.transpose((2, 0, 1))
         sample = {'image': torch.from_numpy(img), 'age': self.ages[idx], 'gender': self.genders[idx]}
         return sample

if __name__ == '__main__':
    ds = utkface("dataset/UTKFace")