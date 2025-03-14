import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2 as cv

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

class AVADataset(Dataset):
    def __init__(self, path_to_csv, images_path,if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path =  images_path
        if if_train:
            self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            normalize])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item] # 从df中获取行数据并存储在row中
        scores_names = [f'score{i}' for i in range(2,12)]
        y = np.array([row[k] for k in scores_names])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        try:
            image = default_loader(image_path)  # 读取为Image对象
        except OSError:
            print(f"\nWarning: Image {image_path} is truncated or doesnt exist. Skip instead.")
            # image = Image.new('RGB', (224, 224), (255, 255, 255))  # 创建一个白色图像
            return None  # 跳过此图像
        x = self.transform(image)
        return x, p.astype('float32')
    

# BAID数据集
mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]

class BBDataset(Dataset):
    def __init__(self, file_dir='dataset', type='train', test=False):
        self.if_test = test
        self.train_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.test_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.images = []
        self.pic_paths = []
        self.labels = []

        if type == 'train':
            DATA = pd.read_csv(os.path.join(file_dir,'dataset','train_set.csv'))
        elif type == 'validation':
            DATA = pd.read_csv(os.path.join(file_dir,'dataset','val_set.csv'))
        elif type == 'test':
            DATA = pd.read_csv(os.path.join(file_dir,'dataset','test_set.csv'))

        labels = DATA['score'].values.tolist()
        pic_paths = DATA['image'].values.tolist()
        for i in tqdm(range(len(pic_paths))):
            pic_path = os.path.join(file_dir,'images', pic_paths[i])
            label = float(labels[i] / 10)
            self.pic_paths.append(pic_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.pic_paths)

    def __getitem__(self, index):
        pic_path = self.pic_paths[index]
        try:
            # 尝试读取图像
            img = cv.imread(pic_path)
            if img is None:
                raise ValueError(f"Failed to load image: {pic_path}")
                
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            # 应用转换
            if self.if_test:
                img = self.test_transformer(img)
            else:
                img = self.train_transformer(img)
                
            return img, self.labels[index]
            
        except Exception as e:
            print(f"\nWarning: Image {pic_path} error: {str(e)}. Skip instead.")
            
            # 策略1: 返回None，需要在DataLoader中使用collate_fn处理
            return None
    

# 读入预测用数据
class TestDataset(Dataset):
    def __init__(self, path_to_csv, images_path):
        self.df = pd.read_csv(path_to_csv)
        self.images_path =  images_path
        
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item] # 从df中获取行数据并存储在row中
        scores_names = [f'score{i}' for i in range(2,12)]
        y = np.array([row[k] for k in scores_names])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)  # 读取为Image对象
        x = self.transform(image)
        return x, p.astype('float32')