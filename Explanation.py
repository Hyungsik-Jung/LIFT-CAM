import torch, torchvision
import torch.nn.functional as F
import torch.nn as nn
from utils import min_max_normalize
from captum.attr import DeepLift, LRP, Lime

# The implementations of LIFT-CAM, LRP-CAM, and LIME-CAM
class CAM_Explanation:
    def __init__(self, model, method):
        self.model = model
        self.method = method
    def __call__(self, x, class_id, image_size):
        if torch.cuda.is_available():
            x = x.cuda()
        if self.method == "LIFT-CAM":
            explanation = lift_cam(self.model, x, class_id)
            with torch.no_grad():
                explanation = F.interpolate(explanation, size=image_size[::-1], mode="bilinear")
        elif self.method == "LRP-CAM":
            explanation = lrp_cam(self.model, x, class_id)
            with torch.no_grad():
                explanation = F.interpolate(explanation, size=image_size[::-1], mode="bilinear")
        elif self.method == "LIME-CAM":
            explanation = lime_cam(self.model, x, class_id)
            with torch.no_grad():
                explanation = F.interpolate(explanation, size=image_size[::-1], mode="bilinear")
        else:
            raise Exception("Not supported method.")
                
        explanation = explanation.detach().cpu()
        explanation = min_max_normalize(explanation)
        return explanation

# The later part of a given original prediction model (i.e. F in the main paper)
class Model_Part(nn.Module):
    def __init__(self, model):
        super(Model_Part, self).__init__()
        self.model_type = None
        if isinstance(model, torchvision.models.vgg.VGG):
            self.model_type = "vgg16"
            self.max_pool = model.features[-1:]
            self.avg_pool = model.avgpool
            self.classifier = model.classifier
        elif isinstance(model, torchvision.models.resnet.ResNet):
            self.model_type = "resnet50"
            self.avg_pool = model.avgpool
            self.classifier = model.fc
    def forward(self, x):
        if self.model_type is "vgg16":
            x = self.max_pool(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)         
        return x
    
# LIFT-CAM
def lift_cam(model, x, class_id=None):
    
    act_map_list = []

    def forward_hook(module, input, output):
        act_map_list.append(output)
    
    if isinstance(model, torchvision.models.vgg.VGG):
        handle = model.features[-2].register_forward_hook(forward_hook)
    elif isinstance(model, torchvision.models.resnet.ResNet):
        handle = model.layer4.register_forward_hook(forward_hook)
    else:
        raise Exception("Not supported architecture.")
        
    output = model(x)
    
    if class_id is None:
        class_id = torch.argmax(output, dim=1)
    act_map = act_map_list[0]
    handle.remove()
    del act_map_list

    model_part = Model_Part(model)
    model_part.eval()
    
    dl = DeepLift(model_part)
    ref_map = torch.zeros_like(act_map).cuda()
    dl_contributions = dl.attribute(act_map, ref_map, target=class_id, return_convergence_delta=False)

    scores_temp = torch.sum(dl_contributions,(2,3),keepdim=False)
    scores = torch.squeeze(scores_temp,0)
    scores = scores.cpu()
    
    vis_ex_map = (scores[None, :, None, None] * act_map.cpu()).sum(dim=1, keepdim=True)
    vis_ex_map = F.relu(vis_ex_map).float()
    
    return vis_ex_map

# LRP-CAM
def lrp_cam(model, x, class_id=None):
    
    act_map_list = []

    def forward_hook(module, input, output):
        act_map_list.append(output)
    
    if isinstance(model, torchvision.models.vgg.VGG):
        handle = model.features[-2].register_forward_hook(forward_hook)
    elif isinstance(model, torchvision.models.resnet.ResNet):
        handle = model.layer4.register_forward_hook(forward_hook)
    else:
        raise Exception("Not supported architecture.")
        
    output = model(x)
    
    if class_id is None:
        class_id = torch.argmax(output, dim=1)
    act_map = act_map_list[0]
    handle.remove()
    del act_map_list

    model_part = Model_Part(model)
    model_part.eval()
    
    lrp = LRP(model_part)
    lrp_contributions = lrp.attribute(act_map, target=class_id, return_convergence_delta=False)

    scores_temp = torch.sum(lrp_contributions,(2,3),keepdim=False)
    scores = torch.squeeze(scores_temp,0)
    scores = scores.cpu()
    
    vis_ex_map = (scores[None, :, None, None] * act_map.cpu()).sum(dim=1, keepdim=True)
    vis_ex_map = F.relu(vis_ex_map).float()
    
    return vis_ex_map

# LIME-CAM
def lime_cam(model, x, class_id=None):
    
    act_map_list = []

    def forward_hook(module, input, output):
        act_map_list.append(output)
    
    if isinstance(model, torchvision.models.vgg.VGG):
        handle = model.features[-2].register_forward_hook(forward_hook)
    elif isinstance(model, torchvision.models.resnet.ResNet):
        handle = model.layer4.register_forward_hook(forward_hook)
    else:
        raise Exception("Not supported architecture.")
        
    output = model(x)
    
    if class_id is None:
        class_id = torch.argmax(output, dim=1)
    act_map = act_map_list[0]
    num_of_channel = int(act_map.size(1))
    handle.remove()
    del act_map_list

    model_part = Model_Part(model)
    model_part.eval()
    
    lime = Lime(model_part)
    ref_map = torch.zeros_like(act_map).cuda()
    f_mask = torch.zeros_like(act_map).long().cuda()
    for i in range(f_mask.size(1)):
      f_mask[:,i,:,:] = i
    lime_contributions = lime.attribute(act_map, baselines=ref_map, n_samples = num_of_channel, 
    target=class_id, feature_mask=f_mask, return_input_shape=False)

    scores = torch.squeeze(lime_contributions,0)
    scores = scores.cpu()
    
    vis_ex_map = (scores[None, :, None, None] * act_map.cpu()).sum(dim=1, keepdim=True)
    vis_ex_map = F.relu(vis_ex_map).float()
    
    return vis_ex_map