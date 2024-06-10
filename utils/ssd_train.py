# TODO: linting
"""
Train SSD model on Agave HD dataset.
"""
import torch
import albumentations as A
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.agave_dataset import AgaveDataset, get_transform
from utils.ssd_model import SSD
from utils.ssd_loss import SSDLoss
from utils.anchors import AnchorUtils
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scales = [6, 3, 1]
centers = [(0.5, 0.5)]
size_scales = [0.5]
aspect_ratios = [(1., 1.), (1.5, 0.8), (1.8, 0.4)]
sizes = [(s * a[0], s * a[1]) for s in size_scales for a in aspect_ratios]
k, anchors, grid_size = AnchorUtils.generate_anchors(scales, centers, sizes)

trans = A.Compose([
    A.Resize(100, 100)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def fit(model, X, target, epochs=1, lr=1e-4):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = SSDLoss(anchors.to(device), grid_size.to(device))
    for epoch in range(1, epochs+1):
        model.train()
        train_loss_loc, train_loss_cls = [], []
        optimizer.zero_grad()
        outputs = model(X.to(device))
        loss = criterion(outputs, (target[0].to(device), target[1].to(device)))
        loss.backward()
        optimizer.step()
        train_loss_loc.append(loss.item())
        print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss_loc):.5f}")
        

if __name__ == "__main__":
    root_dir = 'data/datasets/agaveHD-1'
    dataset = AgaveDataset(root=root_dir, image_set='train', transform=get_transform(train=True))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for images, targets in dataloader:
        break

    img_np = images[0].permute(1, 2, 0).numpy() * 255
    labels, bbs = targets['labels'][0].numpy(), targets['boxes'][0].numpy()

    bbs = [AnchorUtils.norm(bb, img_np.shape[:2]) for bb in bbs]
    labels = [0 if label != 1 else 1 for label in labels]

    augmented = trans(**{'image': img_np / 255.0, 'bboxes': bbs, 'labels': labels})

    img, bbs, labels = augmented['image'], augmented['bboxes'], augmented['labels']

    classes = ["background", "agave"]
    AnchorUtils.plot_anchors(img, (labels, bbs), anchors, classes)
    plt.show()

    img_tensor = torch.FloatTensor(img / 255.).permute(2, 0, 1).unsqueeze(0).to(device)
    bb_tensor = torch.FloatTensor(bbs).unsqueeze(0).to(device)
    label_tensor = torch.tensor(labels).long().unsqueeze(0).to(device)

    model = SSD(n_classes=len(classes), k=k)
    fit(model, dataloader, epochs=100)

    torch.save(model.state_dict(), 'ssd_model.pth')
    print(f"Modelo guardado en 'ssd_model.pth', con k={k}")
