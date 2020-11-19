import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

random.seed(1)

class handsDataset(Dataset):
    def __init__(self, data_dir, transform=True):

        self.transform = transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor()
                                    ])
        #print(data_dir)
        self.data_dir = data_dir
        self.data_info = self.get_img_info()

    def __getitem__(self, item):
        path_img, label = self.data_info[item]

        img = Image.open(path_img).convert('RGB')
        return self.transform(img), label

    def __len__(self):
        return len(self.data_info)

    #@staticmethod
    def get_img_info(self):
        data_info = []

        for root, dirs, _ in os.walk(self.data_dir):
            #遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                #print(sub_dir)
                img_names = list(filter(lambda x: x.endswith(".jpg"), img_names))
                #print(img_names)
                for i in range(len(img_names)):
                    #print(i)
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = sub_dir
                    data_info.append((path_img, int(label)))

        return data_info