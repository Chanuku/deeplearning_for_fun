import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import pdb
from onehot import onehot
import torch
import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class BagDataset(Dataset):

    def __init__(self, transform=None):
       self.transform = transform 
    def __len__(self):
       return len(os.listdir('last'))

    def _convert2labelImage(self, mask):
        '''
        r = mask[:,:,0] >= 128
        #print (r.shape)
        g = ~r & (mask[:,:,1] >= 128)
        b = (~(r | g)) & (mask[:,:,2] >= 128)
        label_img = 1*r + 2*g + 3*b
        return label_img.astype(np.int)
        '''
        
        y = (mask[:,:,0] >= 128) & (mask[:,:,1] >= 128) & (mask[:,:,2] < 128)
        p = ~y & (mask[:,:,0] >= 128) & (mask[:,:,1] < 128) & (mask[:,:,2] >= 128)
        s = (~(y | p)) & (mask[:,:,0] < 128) & (mask[:,:,1] >= 128) & (mask[:,:,2] >= 128)
        r = (~(y | p | s)) & (mask[:,:,0] >= 128) & (mask[:,:,1] < 128) & (mask[:,:,2] < 128)
        g = (~(y | p | s | r)) & (mask[:,:,0] < 128) & (mask[:,:,1] >= 128) & (mask[:,:,2] < 128)
        b = (~(y | p | s | r | g)) &(mask[:,:,0] < 128) & (mask[:,:,1] < 128) & (mask[:,:,2] >= 128)

        label_img = 1*r + 2*g + 3*b + 4*y + 5*p + 6*s
        #f = codecs.open("log.txt","w")
        #print(label_img.shape)
        #f.write(str(label_img.tolist()))
        return label_img.astype(np.int)


    def __getitem__(self, idx):
        img_name = os.listdir('last')[idx]
        imgA = cv2.imread('last/'+img_name)
        imgA = cv2.resize(imgA, (160, 160))
        imgB = cv2.imread('last_msk/'+img_name, 0)
        imgB = cv2.resize(imgB, (160, 160))
        #imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = self._convert2labelImage(imgB)

        #imgB = onehot(imgB, 2)
        imgB = imgB.swapaxes(0, 2).swapaxes(1, 2)
        imgB = torch.FloatTensor(imgB)
        #print(imgB.shape)
        
        if self.transform:
            imgA = self.transform(imgA)    
        item = {'A':imgA, 'B':imgB}
        return item


        

bag = BagDataset(transform)

dataloader = DataLoader(bag, batch_size=4, shuffle=True, num_workers=4)
if __name__ =='__main__':
    for batch in dataloader:
        break
