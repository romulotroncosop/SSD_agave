#%% Load dataset and annotations
from utils.agave_dataset import AgaveDataset, get_transform, get_sample, show_image
from utils.anchors import AnchorUtils
from utils.ssd_model import SSD
from utils.ssd_loss import SSDLoss
from utils.ssd_train import fit
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = AgaveDataset(root='./data/datasets/agaveHD-1', image_set='train', transform=get_transform(train=False))
img, target = dataset[6]
show_image(img, target['boxes'], target['labels'])

img_np, anns = get_sample(dataset, 6)
scales = [6, 3, 1]
centers = [(0.5, 0.5)]
size_scales = [0.5]
aspect_ratios = [(1., 1.), (1.5, 0.8), (1.8, 0.4)]
sizes = [(s * a[0], s * a[1]) for s in size_scales for a in aspect_ratios]
k, anchors, grid_size = AnchorUtils.generate_anchors(scales, centers, sizes)
classes = ["background", "agave"]
AnchorUtils.plot_anchors(img_np, anns, anchors, classes)
print("Número de anchors generados:", len(anchors))
print("Detalle de anchors por escala:", k)
plt.show()

#%% Training the model
dataset = AgaveDataset(root='./data/datasets/agaveHD-1', image_set='train', transform=get_transform(train=True))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Configuración de las transformaciones
trans = A.Compose([
    A.Resize(100, 100)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Obtener y visualizar una muestra de datos
for images, targets in dataloader:
    img_np = images[0].permute(1, 2, 0).numpy() * 255
    labels, bbs = targets['labels'][0].numpy(), targets['boxes'][0].numpy()
    bbs = [AnchorUtils.norm(bb, img_np.shape[:2]) for bb in bbs]
    labels = [0 if label != 1 else 1 for label in labels]
    augmented = trans(**{'image': img_np / 255.0, 'bboxes': bbs, 'labels': labels})
    img, bbs, labels = augmented['image'], augmented['bboxes'], augmented['labels']
    img_tensor = torch.FloatTensor(img / 255.).permute(2, 0, 1).unsqueeze(0).to(device)
    bb_tensor = torch.FloatTensor(bbs).unsqueeze(0).to(device)
    label_tensor = torch.tensor(labels).long().unsqueeze(0).to(device)
    break

print("Imagen Tensor Shape:", img_tensor.shape)
print("Cajas Tensor Shape:", bb_tensor.shape)
print("Etiquetas Tensor Shape:", label_tensor.shape)

# Ajustar el modelo y entrenar
n_classes = len(classes)
k_values = [3, 3, 3]
model = SSD(n_classes=n_classes, k=k_values)
fit(model, dataloader, epochs=100)

# Guardar el modelo
torch.save(model.state_dict(), 'ssd_model.pth')
print(f"Modelo guardado en 'ssd_model.pth', con k={k}")
