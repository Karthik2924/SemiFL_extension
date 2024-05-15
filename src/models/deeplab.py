import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, make_batchnorm#, loss_fn
from config import cfg

#model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=False)

def loss_fn(output, target, reduction='mean'):
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        loss = F.mse_loss(output, target, reduction=reduction)
    return loss


class Deeplab_model(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', weights='DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1')#.to(device)
        #self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large',weights = 'DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1').to(device)
        #self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        #self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        for module in self.model.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    
    def f(self, x):
        #print(x.shape)
        if x.shape[0]==1:
            return self.model(torch.cat((x,x),0))['out'][0].unsqueeze(0)
        return self.model(x)['out']
    def forward(self, input):
        output = {}
        output['target'] = self.f(input['data'])
        # print("***********")
        # print(input['data'].shape,input['target'].shape,output['target'].shape)
        # print("*************")
        # for k in input:
        #     print(f" {k} : {input[k].shape}")
        if 'loss_mode' in input:
            if 'sup' in input['loss_mode']:
            #if input['loss_mode'] == 'sup':
                output['loss'] = loss_fn(output['target'], input['target'])
            elif ('fix' in input['loss_mode']) and ('mix' not in input['loss_mode']):
            #elif input['loss_mode'] == 'fix':
                aug_output = self.f(input['aug'])
                output['loss'] = loss_fn(aug_output, input['target'].detach())
                
            elif 'fix' in input['loss_mode'] and 'mix' in input['loss_mode']:
            # elif input['loss_mode'] == 'fix-mix':
                aug_output = self.f(input['aug'])
                output['loss'] = loss_fn(aug_output, input['target'].detach())
                mix_output = self.f(input['mix_data'])
                output['loss'] += input['lam'] * loss_fn(mix_output, input['mix_target'][..., 0].detach()) + (
                        1 - input['lam']) * loss_fn(mix_output, input['mix_target'][..., 1].detach())
        else:
            if not torch.any(input['target'] == -1):
                output['loss'] = loss_fn(output['target'], input['target'])
        output['mask'] = torch.argmax(output['target'],1)
        return output
    

def deeplab(momentum=None, track=False):
    # data_shape = cfg['data_shape']
    # target_size = cfg['target_size']
    # hidden_size = cfg['resnet18']['hidden_size']
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=False)
    model = Deeplab_model()
    #model.apply(init_param)
    #model.model.classifier.apply(init_param)
    #model.classifier.apply(init_param)
    #model.model.classifier.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))

    return model
