# utils/infer.py
# utils/infer.py

import torch
from utils.ssd_model import SSD

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializar y cargar el modelo
model = SSD(n_classes=2, k=[1, 1, 1])  # Asegúrate de que n_classes y k sean los mismos
model.load_state_dict(torch.load('ssd_model.pth', map_location=device))
model.to(device)
model.eval()

# Ejemplo de uso
img_tensor = torch.rand((1, 3, 100, 100)).to(device)  # Reemplaza con una imagen real
with torch.no_grad():
    pred_bboxes, pred_classes = model(img_tensor)
print("Predicciones de Cajas:", pred_bboxes)
print("Predicciones de Clases:", pred_classes)
