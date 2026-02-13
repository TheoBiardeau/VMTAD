import torch
from torch import nn


class CosineLoss(nn.Module):
    
    def __init__(self, reduction = 'none'):
        super(CosineLoss, self).__init__()
        self.reduction = reduction
        self.cos_dist = torch.nn.CosineSimilarity()
        
    def forward(self,tensor_1, tensor_2): 
        # Calculer la distance cosinus entre les tenseurs normalisés
        cosine_similarity = self.cos_dist(tensor_1, tensor_2)

        # La distance cosinus est définie comme 1 - similarité cosinus
        cosine_distance = 1 - cosine_similarity

        # Calculer la perte moyenne
        if self.reduction == 'none' :
            return(cosine_distance)
            
        elif self.reduction == 'mean':
            return(torch.mean(cosine_distance))
        
        elif self.reduction == 'sum':
            return(torch.sum(cosine_distance))
        


class Gaussian(torch.nn.Module):
    def __init__(self,device, channels=1, kernel_size=3, alpha=1):
        super(Gaussian, self).__init__()
        self.alpha = alpha
        self.kernel_size = kernel_size
        
        # Create a 2D grid for the kernel
        offset = (kernel_size - 1) / 2

        coords = torch.arange(kernel_size, device=device) - offset


        x, y = torch.meshgrid(coords, coords, indexing='xy')
    
        # Calculate the Gaussian filter
        gaussian_filter = self.gaussian_2d(x, y)
        
        # Reshape and repeat the filter for different channels
        self.filter = gaussian_filter.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1).to(device)

    def gaussian_2d(self, x, y, mean_x=0, mean_y=0, std_x=1, std_y=1):
        return (1. / (2 * torch.pi * std_x * std_y)) * \
               torch.exp(-((x - mean_x)**2 / (2 * std_x**2) + (y - mean_y)**2 / (2 * std_y**2)))

    def forward(self, input_frames):
        # Pad the input frames to handle edge effects
        padded_frames = torch.nn.functional.pad(input_frames, 
                                                (self.kernel_size // 2, self.kernel_size // 2,
                                                 self.kernel_size // 2, self.kernel_size // 2))
        
        # Apply the Gaussian filter via convolution
        smoothed_frames = torch.nn.functional.conv2d(padded_frames, self.filter)
        
        # Return the alpha-powered smoothed frames
        return (smoothed_frames ** self.alpha)
