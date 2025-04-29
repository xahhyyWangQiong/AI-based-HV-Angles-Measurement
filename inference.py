import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from ultralytics import YOLO
from postprocessing import line_regression
from matplotlib import pyplot as plt

# Configuration
class Config:
    test_img_dir = 'images'
    seg_model_path = 'model/best_model.pth'
    num_classes = 3
    cmap_classes = 4  # Number of classes for the colormap
    alpha = 0.5  # Transparency for blending

# Transformation pipeline
def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_segmentation_model(seg_model_path, num_classes):
    model = deeplabv3_mobilenet_v3_large(pretrained=False, weights_backbone=False, num_classes=num_classes)
    model.load_state_dict(torch.load(seg_model_path))
    model.eval()
    return model

# Generate VOC colormap
def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype='float32' if normalized else 'uint8')
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << (7 - j))
            g = g | (bitget(c, 1) << (7 - j))
            b = b | (bitget(c, 2) << (7 - j))
            c = c >> 3
        cmap[i] = [r, g, b]
    return cmap / 255 if normalized else cmap

# Process a single image
def process_image(image_path, segmentation_model, transform, config):
    input_image = cv2.imread(image_path)
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image_pil = Image.open(image_path)

    cmap = voc_cmap(N=config.cmap_classes)


    width, height = input_image_pil.size
    scale = 512 / min(width, height)
    resized_image = input_image_pil.resize((int(width * scale), int(height * scale)))

    seg_input = transform(resized_image).unsqueeze(0)
    with torch.no_grad():
        seg_output = segmentation_model(seg_input)['out']
        seg_output = torch.sigmoid(seg_output)
        seg_output_np = seg_output.numpy()[0, :, :, :]
        seg_fit_vis, hva, ima = line_regression(input_image_rgb, [0, 0, width, height], seg_output_np)
    


# Main pipeline
def main():
    config = Config()
    transform = get_transform()
    segmentation_model = load_segmentation_model(config.seg_model_path, config.num_classes)

    for filename in os.listdir(config.test_img_dir):
        image_path = os.path.join(config.test_img_dir, filename)
    
        _, hva, ima = process_image(image_path, segmentation_model, transform, config)


if __name__ == "__main__":
    main()
