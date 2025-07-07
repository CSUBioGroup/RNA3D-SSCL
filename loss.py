import torch.nn as nn
import torch
import math
# from einops import rearrange, einsum 


class ClampMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, max_val):
        ctx.save_for_backward(input)
        ctx.max_val = max_val
        return input.clamp(min=None, max=max_val)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input > ctx.max_val] = grad_output[input > ctx.max_val]
        return grad_input, None

# Usage:



def rigidFrom3Points(x):
    x1,x2,x3 = x[:,0],x[:,1],x[:,2]
    v1=x3-x2
    v2=x1-x2
    e1=v1/(torch.norm(v1,dim=-1,keepdim=True) + 1e-03)
    u2=v2 - e1*(torch.einsum('bn,bn->b',e1,v2)[:,None])
    e2 = u2/(torch.norm(u2,dim=-1,keepdim=True) + 1e-08)
    e3=torch.cross(e1,e2,dim=-1)
    return torch.stack([e1,e2,e3],dim=1),x2[:,:]


class FapeLoss(nn.Module):
    """Fape Loss Function"""
    # def __init__(self, ep=1e-4,epmax=20):
    def __init__(self, ep=1e-3,epmax=30):
        super(FapeLoss, self).__init__()
        self.ep = ep
        self.epmax = epmax

    def forward(self,coor,p_rot,p_tran, target):#L,3,3
        # print(rot.shape)
        # print(trans.shape)
        # print(matrix.shape)

        coor_shape = coor.shape
        coor_new = coor.reshape(-1,coor_shape[-1])
        # print(p_rot.sum())
        # print(p_tran.sum())


        pred_x2 = coor_new[:, None, :] - p_tran[None, :, :]  # Lx Lrot N , 3
        pred_x2 = torch.einsum('ijd,jde->ije', pred_x2,
                                   p_rot.transpose(-1, -2))  # transpose should be equal to inverse

        target_shape = target.shape
        target_new = target.reshape(-1,target_shape[-1])
        t_rot, t_tran = rigidFrom3Points(target)

        tx = target_new[:, None, :] - t_tran[None, :, :]  # Lx Lrot N , 3
        tx = torch.einsum('ijd,jde->ije', tx,
                                   t_rot.transpose(-1, -2))  # transpose should be equal to inverse

        errmap = torch.sqrt((((pred_x2 - tx) ** 2) + self.ep).sum(dim=-1))

            # print(errmap)
        # clamped_errmap = ClampMax.apply(errmap, self.epmax)
        # loss = loss + torch.sum(  clamped_errmap       )
        loss =  torch.mean(  torch.clamp(errmap ,max=self.epmax)   ,dim = (0,1)    ).sum()
        return loss/10.0



class DistanceLoss(nn.Module):
    def __init__(self, distance_bins=torch.linspace(2, 40, 37)):
        """
        Initialize the DistanceLoss module.

        Args:
            distance_bins (torch.Tensor): Bin edges for distances (default 36 intervals from 2 to 40 Å).
        """
        super(DistanceLoss, self).__init__()
        self.register_buffer('distance_bins', distance_bins)
        self.register_buffer('bin_edges', torch.cat([torch.tensor([0.0]), distance_bins, torch.tensor([float('inf')])]))


    def forward(self, true_coor, pred_dis):
        """
        Compute the distance loss (Ldist) as defined in the formula.

        Args:
            true_coor (torch.Tensor): Tensor of shape (L, 3), representing 3D coordinates.
            pred_dis (torch.Tensor): Tensor of shape (L, L, 38), representing predicted probabilities.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        L = true_coor.size(0)
        device = true_coor.device
        mask = torch.ones(L, L, dtype=torch.bool, device=device)
        diag_mask = ~torch.eye(L, L, dtype=torch.bool, device=device)
        mask = mask & diag_mask

        # Step 1: Compute pairwise distances
# Assign distances to bins (one-hot encoding for y_bij)
        dist_matrix = torch.cdist(true_coor, true_coor, p=2)  # Shape: (L, L)
        lower_edges = self.bin_edges[:-1].view(1, 1, -1)  # Shape: (1, 1, num_bins)
        upper_edges = self.bin_edges[1:].view(1, 1, -1)  # Shape: (1, 1, num_bins)        
# Check conditions for all bins at once
        y_bij = ((dist_matrix.unsqueeze(-1) >= lower_edges) & 
         (dist_matrix.unsqueeze(-1) < upper_edges)).float().to(device)  # Shape: (L, L, num_bins)
        y_bij = y_bij * mask.unsqueeze(-1)


        # Step 3: Compute cross-entropy loss
        pred_dis = pred_dis.clamp(min=1e-8, max=1 - 1e-8)  # Avoid log(0)

        loss = -torch.sum(y_bij * torch.log(pred_dis)) / (L * L)
        return loss 



class Fape_pairLoss(nn.Module):
    """Fape Loss Function"""
    def __init__(self, ep=1e-4,epmax=30):
        super(Fape_pairLoss, self).__init__()
        self.ep = ep
        self.epmax = epmax

    def forward(self,coor,rot,trans, target,matrix):#L,3,3
        # print(rot.shape)
        # print(trans.shape)
        # print(matrix.shape)

        p_rot, p_tran = merge_coordinate_systems_batch((rot,trans),matrix)

        coor_shape = coor.shape
        coor_new = coor.view(-1,coor_shape[-1])
        # print(p_rot.sum())
        # print(p_tran.sum())


        pred_x2 = coor_new[:, None, :] - p_tran[None, :, :]  # Lx Lrot N , 3
        pred_x2 = torch.einsum('ijd,jde->ije', pred_x2,
                                   p_rot.transpose(-1, -2))  # transpose should be equal to inverse

        target_shape = target.shape
        target_new = target.view(-1,target_shape[-1])
        t_mean = target[:, [1]]
        t_rot, t_tran = merge_coordinate_systems_batch(rigidFrom3Points(target),matrix)

        tx = target_new[:, None, :] - t_tran[None, :, :]  # Lx Lrot N , 3
        tx = torch.einsum('ijd,jde->ije', tx,
                                   t_rot.transpose(-1, -2))  # transpose should be equal to inverse

        errmap = torch.sqrt((((pred_x2 - tx) ** 2) + self.ep).sum(dim=-1))


            # print(errmap)
        # clamped_errmap = ClampMax.apply(errmap, self.epmax)
        # loss = loss + torch.sum(  clamped_errmap       )
        loss =  torch.mean(  torch.clamp(errmap ,max=self.epmax)   ,dim = (0,1)    ).sum()

        return loss




def merge_coordinate_systems(O_A, O_B, X_A, Y_A, Z_A, X_B, Y_B, Z_B):
    if torch.allclose(O_A, O_B):
        return torch.stack((X_A, Y_A, Z_A)), O_A
    
    O_new = (O_A + O_B) / 2
    
    d = O_B - O_A
    norm_d = torch.norm(d)
    if norm_d == 0:

        return torch.stack((X_A, Y_A, Z_A)), O_A
    X_new = d / norm_d
    
    Z_add = Z_A + Z_B
    Z_sub = Z_A - Z_B
    norm_Z_add = torch.norm(Z_add)
    norm_Z_sub = torch.norm(Z_sub)
    if norm_Z_add == 0 or norm_Z_sub == 0:
        return torch.stack((X_A, Y_A, Z_A)), O_A
    
    if norm_Z_add > norm_Z_sub:
        Z_new = Z_add / norm_Z_add
    else:
        Z_new = Z_sub / norm_Z_sub
    
    Y_new = torch.cross(Z_new, X_new)
    
    if torch.isnan(Y_new).any():
        return torch.stack((X_A, Y_A, Z_A)), O_A
    

    if torch.dot(torch.cross(X_new, Y_new), Z_new) < 0:
        Y_new = -Y_new  
    

    R_new = torch.stack((X_new, Y_new, Z_new))
    
    return R_new, O_new

def merge_coordinate_systems_batch(rot_trans, matrix):
    rot, trans = rot_trans
    L = rot.shape[0]
    new_rots = []
    new_trans = []
    
    for i in range(L):
        for j in range(i+1, L):
            if matrix[i, j] == 1:
                O_A = trans[i]
                O_B = trans[j]
                X_A, Y_A, Z_A = rot[i]
                X_B, Y_B, Z_B = rot[j]
                
                R_new, O_new = merge_coordinate_systems(O_A, O_B, X_A, Y_A, Z_A, X_B, Y_B, Z_B)
                
                new_rots.append(R_new)
                new_trans.append(O_new)
    
    if new_rots:
        new_rots = torch.stack(new_rots)
        new_trans = torch.stack(new_trans)
        new_rot = torch.cat((rot, new_rots), dim=0)
        new_trans = torch.cat((trans, new_trans), dim=0)
    else:
        new_rot = rot
        new_trans = trans
    
    return new_rot, new_trans



    
class Fape_Loss_pair(nn.Module):
    """Fape Loss Function"""
    # def __init__(self, ep=1e-4,epmax=20):
    def __init__(self, ep=1e-3,epmax=40,ep_pair=5,pair_lamda=1):
        super(Fape_Loss_pair, self).__init__()
        self.ep = ep
        self.epmax = epmax
        self.ep_pair = ep_pair
        self.pair_lamda=  pair_lamda      

    def forward(self,coor,p_rot,p_tran, target,matrix):#L,3,3
        L = p_rot.shape[0]
        pred_coor_flat = coor.reshape(L*3, 3)
        true_coor_flat = target.reshape(L*3, 3)
        t_rot, t_tran = rigidFrom3Points(target)

        # print(p_rot.sum())
        # print(p_tran.sum())
        diff_pred = pred_coor_flat[None] - p_tran[:, None]  # (L, L*3,3)
        x_ij = torch.matmul(diff_pred, p_rot.transpose(1,2))  # (L, L*3,3)
        diff_true = true_coor_flat[None] - t_tran[:, None]  # (L, L*3,3)
        x_ij_true = torch.matmul(diff_true, t_rot.transpose(1,2))

        dist = torch.sqrt(torch.sum((x_ij - x_ij_true)**2, dim=-1) + self.ep)
        clamped_dist = torch.clamp(dist, max=self.epmax)
        fape = torch.mean(clamped_dist)
        p_averaged = concatenate_coordinates(coor,matrix)
        t_averaged = concatenate_coordinates(target,matrix)
        if p_averaged is not None:
            p_rot_add, p_tran_add = rigidFrom3Points(p_averaged)
            t_rot_add, t_tran_add = rigidFrom3Points(t_averaged)
            diff_pred_pair = pred_coor_flat[None] - p_tran_add[:, None]  # (L, L*3,3)
            x_ij_pair = torch.matmul(diff_pred_pair, p_rot_add.transpose(1,2))  # (L, L*3,3)
            diff_true_pair = true_coor_flat[None] - t_tran_add[:, None]  # (L, L*3,3)
            x_ij_true_pair = torch.matmul(diff_true_pair, t_rot_add.transpose(1,2))
            dist_pair = torch.sqrt(torch.sum((x_ij_pair - x_ij_true_pair)**2, dim=-1) + self.ep)
            clamped_dist_pair = torch.clamp(dist_pair, max=self.ep_pair)
            return (fape + self.pair_lamda*torch.mean(clamped_dist_pair))/10.0           
        else:
            return fape/10.0



# class Fape_pairLoss2(nn.Module):
#     """Fape Loss Function"""
#     def __init__(self, ep=1e-3,epmax=40):
#         super(Fape_pairLoss2, self).__init__()
#         self.ep = ep
#         self.epmax = epmax

#     def forward(self,coor,rot,trans, target,matrix):#L,3,3
#         L = rot.shape[0]
#         p_averaged = concatenate_coordinates(coor,matrix)
#         if p_averaged is not None:
#             p_rot_add, p_tran_add = rigidFrom3Points(p_averaged)
#             p_rot = torch.cat([rot, p_rot_add], dim=0)
#             p_tran = torch.cat([trans, p_tran_add], dim=0)
#             coor = torch.cat([coor, p_averaged], dim=0)
#         else:
#             p_rot = rot
#             p_tran = trans           
#         L = coor.shape[0]
#         t_averaged = concatenate_coordinates(target,matrix)
#         if t_averaged is not None:
#             target = torch.cat([target, t_averaged], dim=0)
#         pred_coor_flat = coor.reshape(L*3, 3)
#         true_coor_flat = target.reshape(L*3, 3)
#         t_rot, t_tran = rigidFrom3Points(target)

#         # print(p_rot.sum())
#         # print(p_tran.sum())
#         diff_pred = pred_coor_flat[None] - p_tran[:, None]  # (L, L*3,3)
#         x_ij = torch.matmul(diff_pred, p_rot.transpose(1,2))  # (L, L*3,3)
#         diff_true = true_coor_flat[None] - t_tran[:, None]  # (L, L*3,3)
#         x_ij_true = torch.matmul(diff_true, t_rot.transpose(1,2))

#         dist = torch.sqrt(torch.sum((x_ij - x_ij_true)**2, dim=-1) + self.ep)
#         clamped_dist = torch.clamp(dist, max=self.epmax)
#         return torch.mean(clamped_dist)

# class Fape_pairLoss2(nn.Module):
#     """Fape Loss Function"""
#     def __init__(self, ep=1e-3,epmax=40):
#         super(Fape_pairLoss2, self).__init__()
#         self.ep = ep
#         self.epmax = epmax

#     def forward(self,coor,rot,trans, target,matrix):#L,3,3

#         p_averaged = concatenate_coordinates(coor,matrix)
#         if p_averaged is not None:
#             p_rot_add, p_tran_add = rigidFrom3Points(p_averaged)
#             p_rot = torch.cat([rot, p_rot_add], dim=0)
#             p_tran = torch.cat([trans, p_tran_add], dim=0)
#             coor = torch.cat([coor, p_averaged], dim=0)
#         else:
#             p_rot = rot
#             p_tran = trans           

#         t_averaged = concatenate_coordinates(target,matrix)
#         if t_averaged is not None:
#             target = torch.cat([target, t_averaged], dim=0)
#         coor_shape = coor.shape
#         coor_new = coor.view(-1,coor_shape[-1])

#         # print(p_rot.sum())
#         # print(p_tran.sum())


#         pred_x2 = coor_new[:, None, :] - p_tran[None, :, :]  # Lx Lrot N , 3
#         pred_x2 = torch.einsum('ijd,jde->ije', pred_x2,
#                                    p_rot.transpose(-1, -2))  # transpose should be equal to inverse

#         target_shape = target.shape
#         target_new = target.view(-1,target_shape[-1])
#         t_rot, t_tran = rigidFrom3Points(target)

#         tx = target_new[:, None, :] - t_tran[None, :, :]  # Lx Lrot N , 3
#         tx = torch.einsum('ijd,jde->ije', tx,
#                                    t_rot.transpose(-1, -2))  # transpose should be equal to inverse

#         errmap = torch.sqrt((((pred_x2 - tx) ** 2) + self.ep).sum(dim=-1))


#             # print(errmap)
#         # clamped_errmap = ClampMax.apply(errmap, self.epmax)
#         # loss = loss + torch.sum(  clamped_errmap       )
#         loss =  torch.mean(  torch.clamp(errmap ,max=self.epmax)   ,dim = (0,1)    ).sum()

#         return loss



# def concatenate_coordinates(rna_coords, pair_matrix):
#     rows, cols = torch.where(pair_matrix == 1)
#     mask = rows < cols
#     pairs = torch.stack([rows[mask], cols[mask]], dim=1)
    
#     if pairs.size(0) == 0:
#         return None
    
#     i_coords = rna_coords[pairs[:, 0]]  # (N, 3, 3)
#     j_coords = rna_coords[pairs[:, 1]]
#     averaged = (i_coords + j_coords) / 2
    
    
#     return averaged



def concatenate_coordinates(rna_coords, pair_matrix):
    L = rna_coords.size(0)

    indices = torch.nonzero(pair_matrix)
    valid_pairs = indices[indices[:, 0] < indices[:, 1]]
    
    if valid_pairs.size(0) == 0:
        return None
    
    sorted_indices = torch.sort(valid_pairs[:, 0], stable=True).indices
    sorted_valid_pairs = valid_pairs[sorted_indices]
    
    i_indices = sorted_valid_pairs[:, 0]
    j_indices = sorted_valid_pairs[:, 1]
    insert_coords = (rna_coords[i_indices] + rna_coords[j_indices]) / 2
    K = insert_coords.size(0)
    
    indices_list = []
    insert_idx = 0
    for current_i in range(L):
        indices_list.append(current_i)
        if insert_idx < K and sorted_valid_pairs[insert_idx, 0].item() == current_i:
            indices_list.append(L + insert_idx)
            insert_idx += 1
    
    combined = torch.cat([rna_coords, insert_coords], dim=0)
    new_rna_coords = combined[indices_list]
    
    return new_rna_coords






def random_rotation_matrix(device='cpu'):

    axis = torch.randn(3, device=device)
    axis /= torch.norm(axis)  
    

    angle = torch.rand(1, device=device) * 2 * math.pi
    

    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    

    K = torch.tensor([
        [0,         -axis[2],  axis[1]],
        [axis[2],   0,         -axis[0]],
        [-axis[1],  axis[0],   0]
    ], device=device)
    

    R = (
        cos_theta * torch.eye(3, device=device) + 
        (1 - cos_theta) * torch.outer(axis, axis) + 
        sin_theta * K
    )
    
    return R

def random_translation_vector(scale=1.0, device='cpu'):

    return (torch.rand(3, device=device) * 2 - 1) * scale


def apply_rotation_translation(chain, R, t):

    rotated_chain = torch.matmul(chain, R.T)  

    translated_chain = rotated_chain + t
    
    return translated_chain





