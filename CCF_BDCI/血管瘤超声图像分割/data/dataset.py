r'''
Author       : PiKaChu_wcg
Date         : 2021-09-24 18:20:17
LastEditors  : PiKachu_wcg
LastEditTime : 2021-09-24 19:17:57
FilePath     : /wcg/CCF_BDCI/血管瘤超声图像分割/data/dataset.py
'''


from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from .rle2mask import rle_decode
from PIL import Image
import os

class CCFDataset(Dataset):
    def __init__(self,path,transforms=None,mode="train") -> None:
        super().__init__()
        self.path=path if path[-1]=="/" else path+"/"
        self.path="./"+self.path
        if mode=="train":
            self.df=pd.read_csv(path+"/mask.csv",encoding="gbk")
        else:
            self.imglist=[]
            for _,_,file in os.walk(self.path):
                self.imglist.extend(file)
        self.mode=mode
        self.transforms=transforms
    def __len__(self):
        return self.df.size(0)
    def __getitem__(self, index) :
        totensor=transforms.ToTensor()
        if self.mode=="train":
            size=self.df.iloc[index,1].split(" ")
            size=(int(size[1]),int(size[0]))
            rle=self.df.iloc[index,2]
            mask=rle_decode(rle,size)
            imgname=self.df.iloc[index,0]
            img=Image.open(self.path+imgname)
            img=totensor(img)
            mask=totensor(mask)
            if self.transforms:
                img,mask=self.transforms(img,mask)
            return img,mask
        else:
            imgname=self.imglist[index]
            img = Image.open(self.path + imgname)
            img=totensor(img)
            if self.transforms:
                img=self.transforms(img)
            return img

if __name__=="__main__":
    ds = CCFDataset("CCF_BDCI/血管瘤超声图像分割/data/train")
    # print(ds[0])
    t=ds[0]
    import matplotlib.pyplot as plt
    plt.figure()
    print(t[0].shape)
    print(t[1].shape)
    plt.imshow((t[0].numpy().transpose(1, 2, 0)))
    plt.savefig("exp1.jpg")
    plt.imshow((t[1].numpy()).reshape(t[1].shape[1:]))
    plt.savefig("exp2.jpg")
