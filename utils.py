
import os
import numpy as np
import torch
from config import wordcwd,max_len, device, N_cycle
from torch.nn import functional as F
from ss import Extraxt_ss

#solo RNA 
def extract_solorna_data(pdb_path):
    fasta_file = pdb_path.replace(".pdb",".fasta")

    sequence = ""

    with open(fasta_file, "r") as file:
        for line in file:
            if not line.startswith(">"):
                sequence += line.strip()

    coor_path = pdb_path.replace(".pdb","_coor.npy")
    coordinates_array = np.load(coor_path)


    # Assume ss_feature is a function defined elsewhere that generates secondary structure features
    ss_file = pdb_path.replace(".pdb","ss.npy")
    # ss_feature(fasta_file, ss_file, fasta_file)
    Extraxt_ss(fasta_file,ss_file)
    if len(sequence)>max_len:
        return sequence,coordinates_array,ss_file
    ss = np.load(ss_file)
    return sequence, coordinates_array, ss


def valid_rna(in_dir, out_dir,model):
    pdb_files = [f for f in os.listdir(in_dir) if f.endswith('.pdb') and not f.endswith('pre.pdb')]
    model.eval()
    for pdb_ofile in pdb_files:    
        pdb_file = os.path.join(in_dir, pdb_ofile)
        _, file_name, _= split_file_path(pdb_file)
        output_file = os.path.join(out_dir, file_name)+'pre.pdb'
        seq, _, ss = extract_solorna_data(pdb_file)
        model.eval()
        if len(seq)>max_len:
            continue
        with torch.no_grad():
            basenpy = np.load( os.path.join(wordcwd, 'base.npy')  )
            msa=torch.from_numpy(parse_seq(seq))[None,:]
            msa=torch.cat([msa,msa],0)
            msa=F.one_hot(msa.long(),6).float().to(device)
            ss = torch.FloatTensor(ss).to(device)
            base_x = torch.FloatTensor(Get_base(seq,basenpy) ).to(device)
            predxs,pred_dis = model.predict_structures(msa,ss,base_x,N_cycle)
            # predxs,_ = model.predict_structures(msa,ss,base_x,N_cycle)
            pre_coor, rot, trans = predxs[N_cycle-1]
    
            coor = pre_coor.detach().cpu().numpy()
            outpdb_coor(coor,output_file,seq)






def parse_seq(inseq):
    seqnpy=np.zeros(len(inseq))
    seq1=np.array(list(inseq))
    seqnpy[seq1=='A']=1
    seqnpy[seq1=='G']=2
    seqnpy[seq1=='C']=3
    seqnpy[seq1=='U']=4
    seqnpy[seq1=='T']=4
    return seqnpy


def Get_base(seq,basenpy_standard):
    basenpy = np.zeros([len(seq),3,3])
    seqnpy = np.array(list(seq))
    basenpy[seqnpy=='A']=basenpy_standard[0]
    basenpy[seqnpy=='a']=basenpy_standard[0]

    basenpy[seqnpy=='G']=basenpy_standard[1]
    basenpy[seqnpy=='g']=basenpy_standard[1]

    basenpy[seqnpy=='C']=basenpy_standard[2]
    basenpy[seqnpy=='c']=basenpy_standard[2]

    basenpy[seqnpy=='U']=basenpy_standard[3]
    basenpy[seqnpy=='u']=basenpy_standard[3]

    basenpy[seqnpy=='T']=basenpy_standard[3]
    basenpy[seqnpy=='t']=basenpy_standard[3]
    return basenpy
def todevice(adict,device):
    for aterm in adict.keys():
        adict[aterm]=torch.FloatTensor(adict[aterm]).to(device)
    return adict
def tonumpy(adict):
    for aterm in adict.keys():
        adict[aterm]=adict[aterm].cpu().data. numpy()
    return adict


def split_file_path(file_path):
    # 分离目录名和文件名
    dir_name, file_name = os.path.split(file_path)
    # 分离文件名和后缀
    file_base, file_ext = os.path.splitext(file_name)
    return dir_name, file_base, file_ext

def outpdb_coor(coor_np,savefile,seq,start=0,end=1000,energystr=''):
    #rama=torch.cat([rama.view(self.L,2),self.betas],dim=-1)
    Atom_name=[' P  '," C4'",' N1 ']
    last_name=['P','C','N']
    wstr=[f'REMARK {str(energystr)}']
    templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
    count=1
    L = len(seq)
    for i in range(L):
        if seq[i] in ['a','g','A','G']:
            Atom_name = [' P  '," C4'",' N9 ']
            #atoms = ['P','C4']

        elif seq[i] in ['c','u','C','U']:
            Atom_name = [' P  '," C4'",' N1 ']
        for j in range(coor_np.shape[1]):
            outs=('ATOM  ',count,Atom_name[j],seq[i],'A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],0,0,last_name[j],'')
                #outs=('ATOM  ',count,Atom_name[j],'ALA','A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],1.0,90,last_name[j],'')
                #print(outs)
            if i>=start-1 and i < end:
                wstr.append(templet % outs)
            count+=1
            
    wstr='\n'.join(wstr)
    wfile=open(savefile,'w')
    wfile.write(wstr)
    wfile.close()