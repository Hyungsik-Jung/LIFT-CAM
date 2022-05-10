import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.models as models

# VGG16 or ResNet50 pretrained on ImageNet classification are provided.
def load_model(model_archi, ckpt_path):
    if model_archi == "vgg16":
        if ckpt_path is None:
            model = models.vgg16(pretrained=True)
        else:
            model = models.vgg16(pretrained=False)
            model.load_state_dict(torch.load(ckpt_path))
    elif model_archi == "resnet50":
        if ckpt_path is None:
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet50(pretrained=False)
            model.load_state_dict(torch.load(ckpt_path))         
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    for p in model.parameters():
        p.requires_grad = False
    return model

# Min_max_normalization function for torch tensor
def min_max_normalize(vis_ex_map, map_min=0., map_max=1.):
    lim = [vis_ex_map.min(), vis_ex_map.max()]
    if lim[0] != lim[1]:
        vis_ex_map = (vis_ex_map - lim[0]) / (lim[1] - lim[0])
    if map_min != 0. or map_max != 1.:
        vis_ex_map = torch.clamp(vis_ex_map, map_min, map_max)
        
    return vis_ex_map

# Return preprocessed tensors for ImageNet
def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if torch.cuda.is_available():
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad = False)

# Visualize "visual explanation map"
def visualize(original_img, vis_ex_map, method):
    h,w = original_img.shape[:2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Input image")
    vis_ex_map_numpy = torch.squeeze(vis_ex_map).cpu().detach().numpy()
    vis_ex_map_numpy_resized = cv2.resize(vis_ex_map_numpy,(w,h))
    ax2.imshow(vis_ex_map_numpy_resized,cmap="jet")
    ax2.set_title("Visual explanation map")
    fig.suptitle(method,fontweight ="bold")
    return
