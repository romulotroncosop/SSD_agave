# utils/agave_dataset.py
import os
import torch
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as T

class AgaveDataset(VisionDataset):
    def __init__(self, root, image_set='train', transform=None):
        super(AgaveDataset, self).__init__(root, transform=transform)
        self.image_set = image_set
        self.images = []
        self.annotations = []
        
        image_dir = os.path.join(root, image_set)
        annotation_dir = os.path.join(root, image_set)
        
        for img_name in os.listdir(image_dir):
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                self.images.append(os.path.join(image_dir, img_name))
                annotation_name = img_name.replace('.jpg', '.xml').replace('.png', '.xml')
                self.annotations.append(os.path.join(annotation_dir, annotation_name))
    
    def __getitem__(self, index):
        img_path = self.images[index]
        ann_path = self.annotations[index]
        
        img = Image.open(img_path).convert("RGB")
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Assuming all objects are of class 'agave'
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([index])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        
        if self.transform:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.images)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)