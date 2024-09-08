import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.extmath import cartesian
from hausdorff import hausdorff_distance

__all__ = ['Dice loss', 'Cross entropy', 'Focal loss', 'Dice Iou Cross entropy', 'Binary dice loss']


class IOU(nn.Module):
    '''
    Calculate Intersection over Union (IoU) for semantic segmentation.
    
    Args:
        logits (torch.Tensor): Predicted tensor of shape (batch_size, num_classes, height, width, (depth))
        target (torch.Tensor): Ground truth tensor of shape (batch_size, height, width, (depth))
        num_classes (int): Number of classes

    Returns:
        tensor: Mean Intersection over Union (IoU) for the batch.
        list: List of IOU score for each class
    '''
    def __init__(self, num_classes, ignore_index=[0]):
        super(IOU, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
    def forward(self, logits, target):
        pred = logits.argmax(dim=1)        
        target = target.argmax(dim=1)       
        ious = []
        for cls in range(self.num_classes):
            if cls in self.ignore_index: continue
            pred_mask = (pred == cls)
            target_mask = (target == cls)
                            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union == 0: iou = 1.0 
            else: iou = (intersection / union).item()
            ious.append(iou)
        
        mean_iou = sum(ious) / (self.num_classes - len(self.ignore_index))
        return torch.tensor(mean_iou), ious

    
class BinaryDice(nn.Module):
    '''
    Calculate Binary Dice score and Dice loss for binary segmentation or each class in Multiclass segmentation
    
    Args:
        logits (torch.Tensor): Predicted tensor of shape (batch_size, height, width, (depth))
        target (torch.Tensor): Ground truth tensor of shape (batch_size, height, width. (depth))
        
    Returns:
        tensor: Dice score
        tensor: Dice loss
    '''
    def __init__(self, smooth=1e-5, p=2):
        super(BinaryDice, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, logits, target):
        assert logits.shape[0] == target.shape[0], "logits & Target batch size don't match"
        smooth = 1e-5
        intersect = torch.sum(logits * target)        
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(logits * logits)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice
        return dice, loss
        

class Dice(nn.Module):
    '''
    Calculate Dice score and Dice loss for multiclass semantic segmentation
    
    Args:
        output (torch.Tensor): Predicted tensor of shape (batch_size, num_classes, height, width, (depth))
        target (torch.Tensor): Ground truth tensor of shape (batch_size, height, width, (depth))
        num_classes (int): Number of classes 
        
    Returns:
        tensor: Mean dice score over classes
        tensor: Mean dice loss over classes
        list: dice score for each classes
        listL dice loss for each classes
    '''
    def __init__(self, num_classes, weight=None, softmax=True, ignore_index=[0]):
        super(Dice, self).__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.softmax = softmax
        self.ignore_index = ignore_index
        self.binary_dice = BinaryDice()

    def forward(self, logits, target):
        assert logits.shape == target.shape, 'logits & Target shape do not match'
        if self.softmax: logits = F.softmax(logits, dim=1)
        
        DICE, LOSS = 0.0, 0.0
        CLS_DICE, CLS_LOSS = [], []
        for clx in range(target.shape[1]):
            if clx in self.ignore_index: continue
            dice, loss = self.binary_dice(logits[:, clx], target[:, clx])
            CLS_DICE.append(dice.item())
            CLS_LOSS.append(loss.item())
            if self.weight is not None: dice *= self.weights[clx]
            DICE += dice
            LOSS += loss

        num_valid_classes = self.num_classes - len(self.ignore_index)
        return DICE / num_valid_classes, LOSS / num_valid_classes, CLS_DICE, CLS_LOSS
    
    
class WeightedHausdorffDistance(nn.Module):
    def __init__(self, height, width, p=-9, return_2_terms=False, device=torch.device('cuda')):
        '''
        height         (int):  image height
        width          (int):  image width
        return_2_terms (bool): Whether to return the 2 terms
                               of the WHD instead of their sum.
        '''
        super().__init__()
        self.height, self.width = height, width
        self.size = torch.tensor([height, width], dtype=torch.get_default_dtype(), device=device)
        self.max_dist = math.sqrt(height**2 + width**2)
        self.n_pixels = height * width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(height), np.arange(width)]))
        self.all_img_locations = self.all_img_locations.to(device=device, dtype=torch.get_default_dtype())
        self.return_2_terms = return_2_terms
        self.p = p
        
    def _assert_no_grad(self, variables):
        for var in variables:
            assert not var.requires_grad, \
                "nn criterions don't compute the gradient w.r.t. targets - please " \
                "mark these variables as volatile or not requiring gradients"
                
    def cdist(self, x, y):
        '''
        Compute distance between each pair of the two collections of inputs.
        x: Nxd Tensor
        y: Mxd Tensor
        return: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:]
                i.e. dist[i,j] = || x[i,:] - y[j,:] ||
        '''
        difs = x.unsqueeze(1) - y.unsqueeze(0)
        dists = torch.sum(difs**2, -1).sqrt()
        return dists
    
    def generalize_mean(self, tensor, dim, p=-9, keepdim=False):
        assert p < 0
        res= torch.mean((tensor + 1e-6)**p, dim, keepdim=keepdim)**(1./p)
        return res
        
    def forward(self, prob_map, gt, orig_sizes):
        '''
        prob_map: (B x H x W) Tensor of the probability map of the estimation.
                              B is batch size, H is height and W is width.
                              Values must be between 0 and 1.
                              
        gt: List of Tensors of the Ground Truth points.
            Must be of size B as in prob_map.
            Each element in the list must be a 2D Tensor,
            where each row is the (y, x), i.e, (row, col) of a GT point.
        
        orig_sizes: Bx2 Tensor containing the size
                    of the original images.
                    B is batch size.
                    The size must be in (height, width) format. 
                    
        return: Single-scalar Tensor with the Weighted Hausdorff Distance.
                If self.return_2_terms=True, then return a tuple containing
                the two terms of the Weighted Hausdorff Distance.
        '''
        
        self._assert_no_grad(gt)
        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s'\
            % str(prob_map.size())
            
        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)
        
        terms_1 = []
        terms_2 = []
        for b in range(batch_size):
            
            # One by one
            prob_map_b = prob_map[b, :, :]
            gt_b = gt[b]
            orig_size_b = orig_sizes[b, :]
            norm_factor = (orig_size_b / self.size).unsqueeze(0)
            n_gt_pts = gt_b.size()[0]
            
            # Corner case: no GT points
            if gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 0:
                terms_1.append(torch.tensor([0], 
                                            dtype=torch.get_default_dtype()))
                terms_2.append(torch.tensor([self.max_dist],
                                            dtype=torch.get_default_dtype())) 
                continue
            
            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            normalized_x = norm_factor.repeat(self.n_pixels, 1) * self.all_img_locations
            normalized_y = norm_factor.repeat(len(gt_b), 1) * gt_b
            d_matrix = self.cdist(normalized_x, normalized_y)
            
            # Reshape probability map as a long column vector
            # and prepare it for mulitplication
            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)
            
            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + 1e-6)) * torch.sum(p * torch.min(d_matrix, 1)[0])
            weighted_d_matrix = (1 - p_replicated)*self.max_dist + p_replicated*d_matrix
            minn = self.generalize_mean(weighted_d_matrix,
                                  p=self.p,
                                  dim=0, keepdim=False)
            term_2 = torch.mean(minn)

            terms_1.append(term_1)
            terms_2.append(term_2)
            
        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)
        
        if self.return_2_terms: res = terms_1.mean(), terms_2.means()
        else: res = terms_1.mean() + terms_2.mean()
        return res
    

class HD(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, logits, target):
        _,logits = torch.max(logits, dim=1)
        _,target = torch.max(target, dim=1)
        
        logits = logits.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        
        hd = 0
        for index in range(logits.shape[0]):
            hd += hausdorff_distance(logits[index], target[index], distance='euclidean')
        
        return hd / logits.shape[0]