# TODO: linting

"""
End to End training and inference on Agave HD dataset.
"""
#%%
from utils.agave_dataset import AgaveDataset, get_transform, get_sample, classes, plot_anns
from utils.anchors import AnchorUtils
from utils.ssd_loss import actn_to_bb
from utils.ssd_train import fit
from utils.ssd_model import SSD
from utils.ssd_loss import SSDLoss
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
import albumentations as A
from pathlib import Path
from wasabi import Printer

msg = Printer()

# Dataset and annotations
msg.info("Loading dataset and annotations...")
agave_dataset_path = Path('./data/datasets/agaveHD-1')
if not agave_dataset_path.exists():
    msg.fail("Dataset not found. Please download the dataset first.")
    msg.info("You can download the dataset running: python -m utils.agave_dataset, and move the data to ./data/datasets/")
    exit()
dataset = AgaveDataset(root=agave_dataset_path, image_set='train', transform=get_transform(train=False))
idx = 50 # Define a sample index
img_np, anns = get_sample(dataset, idx)
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
msg.info(f"Anchors: {len(anchors)}, k: {k}")

# Model
n_classes = len(classes)  #  [background, agave]
k_values = [3, 3, 3]
#%%
msg.info("SSD criterion and Model...")
net = SSD(n_classes=n_classes, k=k_values).to(device)
input_tensor = torch.rand((64, 3, 100, 100)).to(device)

output = net(input_tensor)
msg.info(output[0].shape, output[1].shape) # -> target: torch.Size([64, 138, 4]) torch.Size([64, 138, 2])


# Loss (SSDLoss)
criterion = SSDLoss(
    anchors=anchors,
    grid_size=grid_size,
    threshold=0.4,
)

targets = (torch.rand((64, 5, 4)).to(device), torch.randint(0, n_classes, (64, 5)).to(device))
loss = criterion(output, targets)
print(f"Loss: {loss.item()}")

# Transformations
trans = A.Compose([
    A.Resize(100, 100)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

labels, bbs = anns
bb_norm = [AnchorUtils.norm(bb, img_np.shape[:2]) for bb in bbs]
augmented = trans(**{'image': img_np, 'bboxes': bb_norm, 'labels': labels})
img, bbs, labels = augmented['image'], augmented['bboxes'], augmented['labels']

img_tensor = torch.FloatTensor(img / 255.).permute(2,0,1).unsqueeze(0).to(device)
bb_norm = [AnchorUtils.norm(bb, img.shape[:2]) for bb in bbs]
bb_tensor = torch.FloatTensor(bb_norm).unsqueeze(0).to(device)
label_tensor = torch.tensor(labels).long().unsqueeze(0).to(device)

# Training the model
model = SSD(n_classes = len(classes), k=k)
model.to(device)
fit(net, img_tensor, (bb_tensor, label_tensor), epochs=500)
# Prediction function
def predict(model, X):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        bbs, labels = model(X)
        bbs = actn_to_bb(bbs[0], anchors, grid_size)
    return bbs, torch.max(torch.softmax(labels, axis=2)[0].cpu(), axis=1)  # Asegúrate de que las etiquetas estén en la CPU

bbs, (scores, labels) = predict(model, img_tensor)
bbs = [AnchorUtils.unnorm(bb.cpu().numpy(), img.shape[:2]) for bb in bbs]
#%%
# Plotting predictions
plot_anns(img, (labels, bbs))
plt.show()
#%%
plot_anns(img, (labels, bbs), bg=0)
plt.show()
#%%
bbs, (scores, labels) = predict(model, img_tensor)
# Filtrar los valores
mask = labels > 0
bbs = bbs[mask]
labels = labels[mask]
scores = scores[mask]

# Mover los tensores filtrados a CUDA
bbs = bbs.to(device)
labels = labels.to(device)
scores = scores.to(device)

nms_ixs = torchvision.ops.nms(bbs, scores, iou_threshold=0.8)

#%%
bbs, labels = bbs[nms_ixs], labels[nms_ixs]
bbs = [AnchorUtils.unnorm(bb.cpu(), img.shape[:2]) for bb in bbs]  # Mueve los tensores a la CPU antes de la conversión
plot_anns(img, (labels.cpu(), bbs))  # Mueve las etiquetas a la CPU también
plt.show()
