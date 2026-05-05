import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

class GradCAM:
    """
    ResNet modelleri için Grad-CAM (Gradient-weighted Class Activation Mapping) üretici.
    Modelin kararı verirken görüntünün hangi bölgesine odaklandığını görselleştirir.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook'ları kaydet
        self.hooks = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients)
        ]

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        """Isı haritası üretir."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        # Ağırlıklı aktivasyon haritası hesapla
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        
        weights = np.mean(gradients, axis=(2, 3))[0]
        cam = np.zeros(activations.shape[2:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]
            
        # ReLU uygula ve normalize et
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            
        return cam

    def __del__(self):
        # Hook'ları temizle
        for hook in self.hooks:
            hook.remove()

def apply_heatmap_to_image(original_image_path, heatmap, alpha=0.4):
    """Isı haritasını orijinal görüntü üzerine bindirir."""
    img = cv2.imread(str(original_image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
    
    heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
    
    # Overlay (Bindirme)
    superimposed_img = heatmap_img * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(superimposed_img)
