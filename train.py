import sys
import numpy as np
import os
import model as model_py
import torch
from loss import FapeLoss,DistanceLoss,Fape_Loss_pair
from torch.nn import functional as F
from torch.utils.data import DataLoader
import random
import numpy as np
import os
from predict_refine import process_arena
from dataset import RNADataset
from utils import parse_seq,Get_base,valid_rna
from result import compute_tm
import math
from config import msa_dim,m_dim,s_dim,z_dim,N_ensemble,N_cycle,wordcwd,data_path,epoch_num,accumulation_steps
from config import save_model_path ,max_len,model_name,device,seed,lr,T_max,save_fren,ara_fren
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)  






def lr_lambda(epoch):
        # adjust learning rate function
    if epoch< 5:
        return 10
    elif epoch< 15:
        return 3                    
    else:
        return 1


def random_sample_rna(seq, coords, ss, sample_len=160):
    """
    Randomly samples a continuous segment of RNA sequence, coordinates, and secondary structure.

    Args:
        seq (str): RNA sequence as a string.
        coords (torch.Tensor): Tensor of shape (L, 3, 3) representing coordinates.
        ss (torch.Tensor): Tensor of shape (L, L, 4) representing secondary structure.
        sample_len (int): Maximum length of the sampled segment (default: 200).

    Returns:
        tuple: Sub-sampled RNA sequence, coordinates, and secondary structure.
    """
    seq = seq[0]
    coords = coords.squeeze(0).float()
    if len(seq)>max_len:
        full_ss = np.load(ss[0])
    else:          
        full_ss = ss.squeeze(0).float()
    # Get the length of the RNA sequence
    L = len(seq)
    if L <= sample_len:
        return seq, coords, full_ss  # No sampling needed
    
    # Randomly choose a starting index
    start_idx = random.randint(0, L - sample_len)
    end_idx = start_idx + sample_len
    
    # Slice the sequence
    sampled_seq = seq[start_idx:end_idx]
    
    # Slice the tensors
    sampled_coords = coords[start_idx:end_idx, :, :]  # Extract a slice along the first dimension
    sampled_ss = full_ss[start_idx:end_idx, start_idx:end_idx, :]  # Extract the corresponding secondary structure
    if len(seq)>max_len:
        sampled_ss = full_ss[start_idx:end_idx, start_idx:end_idx, :].copy()        
        del full_ss
    else:
        sampled_ss = full_ss[start_idx:end_idx, start_idx:end_idx, :]

    return sampled_seq, sampled_coords, sampled_ss


#dis loss
def pipeline():
    # loss = Mse_loss().to(device)
    # loss = FapeLoss().to(device)
    loss3 = FapeLoss().to(device)
    loss2 = DistanceLoss().to(device)
    loss = Fape_Loss_pair().to(device)
    # loss2 = torch.nn.L1Loss().to(device)
    # loss = FapeLoss().to(device)
    rna_dataset_train = RNADataset(os.path.join( data_path,'train'),True)
    rna_dataset_valid = RNADataset(os.path.join( data_path,'valid'),True)
    train_data_loader = DataLoader(rna_dataset_train, batch_size=1, shuffle=True,drop_last=True)
    valid_data_loader = DataLoader(rna_dataset_valid, batch_size=1, shuffle=True,drop_last=True)


    model = model_py.EvolutionaryStructurePredictor(msa_dim-1,msa_dim,N_ensemble,N_cycle, m_dim,s_dim,z_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max)

    basenpy = np.load( os.path.join(wordcwd, 'base.npy')  )
    print("start train")
    sys.stdout.flush()
    
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0                 
        best_model = model
        best_loss = 100000

        for i,(seq, coords, ss) in enumerate(train_data_loader):
            # seq, coords = extract_rna_data(os.path.join(os.path.join(os.getcwd(), 'test'), '1A3M_1_A-B.pdb'))
            # seq = seq[0]
            # coords = coords.squeeze(0).float().to(device)           
            # ss = ss.squeeze(0).float()
            seq, coords, ss = random_sample_rna(seq, coords, ss)
            coords = coords.to(device)
            msa=torch.from_numpy(parse_seq(seq))[None,:]
            msa=torch.cat([msa,msa],0)
            msa=F.one_hot(msa.long(),6).float().to(device)
            ss = torch.FloatTensor(ss).to(device)
            base_x = torch.FloatTensor(Get_base(seq,basenpy) ).to(device)  

            predxs,pred_dis = model.predict_structures(msa,ss,base_x,N_cycle)         
            pre_coor,rot,trans = predxs[N_cycle-1]   
            loss_batch1 = loss(pre_coor,rot,trans,coords,ss[:, :, 0])
            loss_batch2 = loss2(coords[:,1,:],pred_dis)
            loss_batch =  1.5 * loss_batch1 + 0.6 *loss_batch2
            for n in range(N_cycle-1):
                pre_coor,rot,trans = predxs[n]
                loss_batch += 0.5 * loss3(pre_coor,rot,trans,coords)/(N_cycle-1)
            loss_batch = 2 * loss_batch/math.sqrt(len(seq))
            total_loss += loss_batch.item()
            total_loss1 += 1.5 * loss_batch1.item()/math.sqrt(len(seq))
            total_loss2 += 0.6 *loss_batch2.item()/math.sqrt(len(seq))


            # optimizer.zero_grad()
            loss_batch.backward()

            # optimizer.step()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()       
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        if (epoch+1)%20==0:
            model.eval()
            total_valid_loss = 0
            total_valid_loss1 = 0
            total_valid_loss2 = 0        
            with torch.no_grad():
                for seq, coords, ss in valid_data_loader:
                    seq = seq[0]
                    if len(seq)>max_len:
                        continue
                    coords = coords.squeeze(0).float().to(device)
                    ss = ss.squeeze(0).float().to(device)
                    msa = torch.from_numpy(parse_seq(seq))[None, :]
                    msa = torch.cat([msa, msa], 0)
                    msa = F.one_hot(msa.long(), 6).float().to(device)
                    base_x = torch.FloatTensor(Get_base(seq, basenpy)).to(device)
                    predxs,pred_dis = model.predict_structures(msa,ss,base_x,N_cycle)                
                    pre_coor,rot,trans = predxs[N_cycle-1]

                    loss_batch1 = loss(pre_coor,rot,trans,coords,ss[:, :, 0])
                    loss_batch2 = loss2(coords[:,1,:],pred_dis)
                    loss_batch = 2.0 * loss_batch1 + 0.6 * loss_batch2
                    total_valid_loss +=  loss_batch.item()
                    total_valid_loss1 +=  loss_batch1.item()
                    total_valid_loss2 +=  loss_batch2.item()                                 
            avg_valid_loss = total_valid_loss / len(valid_data_loader)
            print(f"Epoch {epoch+1}/{epoch_num},test Loss: {avg_valid_loss},test Loss1: {total_valid_loss1/len(valid_data_loader)},Train Loss2:{total_valid_loss2/len(valid_data_loader)}")  
        print(f"Epoch {epoch+1}/{epoch_num},Train Loss: {total_loss/len(train_data_loader)},Train Loss1: {total_loss1/len(train_data_loader)},Train Loss2:{total_loss2/len(train_data_loader)}")
         

        scheduler.step()  
        print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()}")


        sys.stdout.flush()       

        if (epoch+1)%save_fren==0:
            torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, os.path.join(save_model_path,f'checkpoint_epoch_{epoch+1}{model_name}.pth'))
            
        
        if (epoch+1)%ara_fren==0:
            arena_path = os.path.join(save_model_path,model_name+str(epoch))
            if not os.path.exists(arena_path):
                os.makedirs(arena_path)  
            valid_rna(os.path.join( data_path,'valid'),arena_path,model)
            process_arena(arena_path)
            compute_tm(arena_path, os.path.join( data_path,'valid'))    
            torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, os.path.join(save_model_path,f'checkpoint_epoch_{epoch+1}{model_name}.pth')) 


    torch.save(best_model, os.path.join(save_model_path,'model.pkl'))




    
if __name__ == "__main__":
    pipeline()
