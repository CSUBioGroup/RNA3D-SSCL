import torch
import os


# ========== 配置参数 ==========
#Please configure the corresponding parameters
msa_dim=6+1
m_dim,s_dim,z_dim = 64,64,64
N_ensemble,N_cycle=1,4
wordcwd = os.getcwd() #Please modify the working directory
data_path= os.getcwd() #Please modify the directory where the data is located
save_model_path = os.getcwd() #Please modify the directory where the model is saved
model_name = 'modelname'
device = torch.device("cuda:0" )
seed = 2024
max_len = 1000
lr = 0.001
T_max = 600
save_fren = 50
ara_fren = 100
epoch_num = 600
accumulation_steps = 64
ARENA_BIN_PATH = os.path.join(wordcwd,"Arena")
checkpoint_path = ""
pre_in_dir = os.path.join(os.getcwd(),"valid")
pre_out_dir = os.path.join(os.getcwd(),"out")