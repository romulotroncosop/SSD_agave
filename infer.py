#%%
# utils/infer.py
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as T
from utils.ssd_model import SSD

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializar y cargar el modelo
model = SSD(n_classes=2, k=[3, 3, 3]).to(device)
model.load_state_dict(torch.load('ssd_model.pth', map_location=device))
model.to(device)
model.eval()

# Ruta de la imagen de prueba
#image_path = 'data/datasets/agaveHD-1/valid/DSC00655_geotag_2_jpg.rf.8794a33103e84fc6f2d09ace90af7a7b.jpg'
image_path = 'data/datasets/agaveHD-1/train/DSC00634_geotag_1_jpg.rf.5b889e1201646bf64789d5b4ca3b1a88.jpg'

# Cargar y transformar la imagen
img = Image.open(image_path).convert("RGB")
transform = T.Compose([
    T.Resize((100, 100)),
    T.ToTensor()
])
img_tensor = transform(img).unsqueeze(0).to(device)

# Inferencia
with torch.no_grad():
    pred_bboxes, pred_classes = model(img_tensor)

# Mover tensores a la CPU y convertir a numpy
pred_bboxes = pred_bboxes.cpu().numpy()
pred_classes = pred_classes.cpu().numpy()

# Visualización
fig, ax = plt.subplots(1)
ax.imshow(img)

for bbox, cls in zip(pred_bboxes[0], pred_classes[0]):
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(bbox[0], bbox[1], f'cls {int(cls[0])}', color='r')  # Asegúrate de extraer el primer elemento de cls

plt.show()
