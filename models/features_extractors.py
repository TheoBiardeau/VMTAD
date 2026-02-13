import torch
from torch import nn
import timm

class EfficientNet_feature_B0(nn.Module):
    def __init__(self, config, device='cuda'):
        super().__init__()
        
        self.model = timm.create_model('tf_efficientnet_b0.in1k', features_only=True, pretrained=True).eval().to(device)
        self.resize = torch.nn.Upsample(size=(config['general_config']['feature_size'],
                                             config['general_config']['feature_size']), mode='bilinear')
    def forward(self, input):
        outputs = self.model(input)
        concatenated_tensor = []
        for i, x in enumerate(outputs):
            if i == 0:
                concatenated_tensor = self.resize(x)
            else:
                concatenated_tensor = torch.cat((concatenated_tensor, self.resize(x)), dim=1)
            
            if i == 2:
                break

        return concatenated_tensor
    
class EfficientNet_feature_B5(nn.Module):
    
    def __init__(self,config, device = 'cuda'):
        super().__init__()
        
        self.model = timm.create_model('tf_efficientnet_b5.in1k', features_only=True, pretrained=True).eval().to(device)
        self.resize = torch.nn.Upsample(size=(config['general_config']['feature_size'],
                                             config['general_config']['feature_size']), mode='bilinear')
    def forward(self, input):
        outputs = self.model(input)
        concatenated_tensor = []
        
        for i, x in enumerate(outputs) :
            if i == 0 : 
                concatenated_tensor = self.resize(x)
            else : 
                concatenated_tensor = torch.cat((concatenated_tensor,self.resize(x)), dim=1)
            if i == 2 : 
                break

        return concatenated_tensor
