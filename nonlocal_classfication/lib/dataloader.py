
import os
import torch.utils.data as data
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImgLoader(data.Dataset):
    def __init__(self, root, ann_file, transform=None, target_transform=None):
        print('=> loading annotations from: ' + os.path.basename(ann_file) + ' ...')
        self.root = root
        with open(ann_file, 'r') as f:
            self.imgs = f.readlines()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        ls = self.imgs[index].strip().split()
        img_path = ls[0]
        target = int(ls[1])
        img = Image.open(
                os.path.join(self.root, img_path)).convert('RGB')
        return self.transform(img), target

    def __len__(self):
        return len(self.imgs)

