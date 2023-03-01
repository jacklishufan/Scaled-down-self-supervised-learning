from torchvision.models.segmentation import FCN
from torchvision.models.resnet import resnet50
import torch
#from detectron2.checkpoint import DetectionCheckpointer
import pickle
from torchvision.models._utils import IntermediateLayerGetter
import sys
src = sys.argv[1]
tgt = sys.argv[2]
print(src,tgt)
#exit()
obj = torch.load(src,map_location='cpu')['state_dict']
newmodel = {}

for k in list(obj.keys()):
    # retain only encoder_q up to before the embedding layer
    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc') :
        # remove prefix
        newmodel[k[len("module.encoder_q."):]] = obj[k]
    # delete renamed or unused k


newmodel['fc.weight']=torch.rand(1000,2048)   
newmodel['fc.bias']=torch.rand(1000)   
        
backbone = resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True])
#
return_layers = {"layer4": "out"}
return_layers["layer3"] = "aux"
print(backbone)
#backbone = IntermediateLayerGetter(backbone,)
backbone.load_state_dict(newmodel)
torch.save(backbone.state_dict(),tgt)