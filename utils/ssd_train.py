# utils/train.py

import torch
import albumentations as A
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.agave_dataset import AgaveDataset, get_transform
from utils.ssd_model import SSD
from utils.ssd_loss import SSDLoss
from utils.anchors import AnchorUtils
import numpy as np

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuración de los parámetros de las anclas
scales = [6, 3, 1]
centers = [(0.5, 0.5)]
size_scales = [0.5]
aspect_ratios = [(1., 1.), (1.5, 0.8), (1.8, 0.4)]
sizes = [(s * a[0], s * a[1]) for s in size_scales for a in aspect_ratios]
k, anchors, grid_size = AnchorUtils.generate_anchors(scales, centers, sizes)

# Transformaciones de las imágenes
trans = A.Compose([
    A.Resize(100, 100)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Función de entrenamiento
def fit(model, dataloader, epochs=1, lr=3e-4):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = SSDLoss(anchors.to(device), grid_size.to(device))
    for epoch in range(1, epochs+1):
        model.train()
        train_loss_loc, train_loss_cls = [], []
        for images, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(images.to(device))
            total_loss, loc_loss, clas_loss = criterion(outputs, (targets['boxes'].to(device), targets['labels'].to(device)))
            total_loss.backward()
            optimizer.step()
            train_loss_loc.append(loc_loss.item())
            train_loss_cls.append(clas_loss.item())
        
        print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss_loc):.5f} (Localization: {np.mean(train_loss_loc):.5f}, Classification: {np.mean(train_loss_cls):.5f})")

if __name__ == "__main__":
    # Inicializar el dataset y el modelo
    root_dir = 'data/datasets/agaveHD-1'
    dataset = AgaveDataset(root=root_dir, image_set='train', transform=get_transform(train=True))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Obtener una muestra de datos y visualizarla
    for images, targets in dataloader:
        break

    # Aplicar transformaciones
    img_np = images[0].permute(1, 2, 0).numpy() * 255
    labels, bbs = targets['labels'][0].numpy(), targets['boxes'][0].numpy()

    # Normalizar las cajas
    bbs = [AnchorUtils.norm(bb, img_np.shape[:2]) for bb in bbs]
    print("Labels:", labels)
    # Asegurarse de que todas las etiquetas son 0 (background) o 1 (agave)
    labels = [0 if label != 1 else 1 for label in labels]

    # Aplicar albumentations
    augmented = trans(**{'image': img_np / 255.0, 'bboxes': bbs, 'labels': labels})

    img, bbs, labels = augmented['image'], augmented['bboxes'], augmented['labels']

    # Visualizar anclas
    classes = ["background", "agave"]
    AnchorUtils.plot_anchors(img, (labels, bbs), anchors, classes)
    plt.show()

    # Preparar tensores para el entrenamiento
    img_tensor = torch.FloatTensor(img / 255.).permute(2, 0, 1).unsqueeze(0).to(device)
    bb_tensor = torch.FloatTensor(bbs).unsqueeze(0).to(device)
    label_tensor = torch.tensor(labels).long().unsqueeze(0).to(device)

    print("Imagen Tensor Shape:", img_tensor.shape)
    print("Cajas Tensor Shape:", bb_tensor.shape)
    print("Etiquetas Tensor Shape:", label_tensor.shape)

    # Inicializar y entrenar el modelo
    model = SSD(n_classes=len(classes), k=k)
    fit(model, dataloader, epochs=100)

    # Guardar el modelo
    torch.save(model.state_dict(), 'ssd_model.pth')
    print(f"Modelo guardado en 'ssd_model.pth', con k={k}")