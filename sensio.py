#%% Load dataset and annotations
#from utils.agave_dataset import AgaveDataset, get_transform, get_sample, show_image
from utils.voc_dataset import get_sample, plot_anns
from utils.voc_dataset import classes
from utils.anchors import AnchorUtils
from utils.ssd_model import SSD
from utils.ssd_loss import SSDLoss
import numpy as np
import matplotlib.pyplot as plt
import torch

# Dataset and annotations
idx = 4445
img_np, anns = get_sample(idx)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#  Anchors
scales = [6, 3, 1]
centers = [(0.5, 0.5)] 
size_scales = [0.5]
aspect_ratios = [(1., 1.), (1.5, 0.8), (1.8, 0.4)]

sizes = [(s * a[0], s * a[1]) for s in size_scales for a in aspect_ratios]

k, anchors, grid_size = AnchorUtils.generate_anchors(scales, centers, sizes)
anchors = anchors.to(device)
grid_size = grid_size.to(device)

AnchorUtils.plot_anchors(img_np, anns, anchors, classes)
plt.show()
print(f"Anchors: {len(anchors)}, k: {k}")


# Model
n_classes = len(classes)
k_values = [3, 3, 3]


net = SSD(n_classes=n_classes, k=k_values).to(device)
input_tensor = torch.rand((64, 3, 100, 100)).to(device)

output = net(input_tensor)
print(output[0].shape, output[1].shape) # -> target: torch.Size([64, 138, 4]) torch.Size([64, 138, 2])


# Loss (SSDLoss)
criterion = SSDLoss(
    anchors=anchors,
    grid_size=grid_size,
    threshold=0.4,
)

targets = (torch.rand((64, 5, 4)).to(device), torch.randint(0, n_classes, (64, 5)).to(device))
#%%
loss = criterion(output, targets)
print(f"Loss: {loss.item()}")

#%% Training the model
import albumentations as A
from utils.anchors import AnchorUtils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trans = A.Compose([
    A.Resize(100, 100)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

img_np, anns = get_sample(idx)
labels, bbs = anns

bbs_normalized = [AnchorUtils.norm(bb, img_np.shape[:2]) for bb in bbs]

augmented = trans(image=img_np, bboxes=bbs_normalized, labels=labels)
img, bbs_aug, labels_aug = augmented['image'], augmented['bboxes'], augmented['labels']

bbs = [AnchorUtils.unnorm(bb, img.shape[:2]) for bb in bbs_aug]

img_tensor = torch.FloatTensor(img / 255.).permute(2, 0, 1).unsqueeze(0).to(device)
bb_tensor = torch.FloatTensor(bbs).unsqueeze(0).to(device)
label_tensor = torch.tensor(labels_aug).long().unsqueeze(0).to(device)

print("Shapes de los tensores:", img_tensor.shape, bb_tensor.shape, label_tensor.shape)

n_classes = len(classes)
k_values = [3, 3, 3]

net = SSD(n_classes=n_classes, k=k_values).to(device)

def fit(model, X, target, epochs=1, lr=1e-4):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = SSDLoss(anchors.to(device), grid_size.to(device))
    for epoch in range(1, epochs+1):
        model.train()
        train_loss_loc, train_loss_cls = [], []
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        train_loss_loc.append(loss.item())
        print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss_loc):.5f}")

# Entrenar el modelo
fit(net, img_tensor, (bb_tensor, label_tensor), epochs=100, lr=1e-3)

#%%
