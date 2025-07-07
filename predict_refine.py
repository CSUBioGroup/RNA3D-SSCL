import os
import glob
import subprocess
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import simtk.openmm as mm
import psutil
import model as model_py
import os
from utils import split_file_path,parse_seq,Get_base,extract_solorna_data,outpdb_coor
from config import wordcwd,max_len,msa_dim, N_ensemble, N_cycle, m_dim, z_dim, s_dim,device
import numpy as np
import torch
from torch.nn import functional as F



ARENA_BIN_PATH = os.path.join(wordcwd,"Arena")
WORKING_DIR = os.path.join(wordcwd,"work")






# ========== 配置参数 ==========
#Please configure the corresponding parameters
checkpoint_path = "/repository/users/lujw/solomodel/checkpoint_epoch_200drnewlosstrain102026.pth"
pre_in_dir = os.path.join(os.getcwd(),"valid")
pre_out_dir = os.path.join(os.getcwd(),"out")




def predict(in_dir, out_dir):
    pdb_files = [f for f in os.listdir(in_dir) if f.endswith('.pdb') and not f.endswith('pre.pdb')]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)  
    model = model_py.EvolutionaryStructurePredictor(msa_dim-1,msa_dim,N_ensemble,N_cycle, m_dim,s_dim,z_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict']) 
    model.load_state_dict(checkpoint)    
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




# Arena处理 
def process_arena(dir):
    """处理PDB文件执行Arena程序"""
    os.makedirs(dir, exist_ok=True)
    
    for pdb_path in glob.glob(os.path.join(dir, '*.pdb')):
        # 跳过已处理文件
        print("Arena "+ pdb_path)
        base_name = os.path.splitext(os.path.basename(pdb_path))[0]
        output_path = os.path.join(dir, f"{base_name}prefull.pdb")
        
        if os.path.exists(output_path):
            continue
        
                
        # 执行Arena程序
        subprocess.run([
            ARENA_BIN_PATH,
            pdb_path,
            output_path,
            '6'  # 假设这是Arena的参数
        ], check=True,
        stdout=subprocess.DEVNULL,  # 忽略标准输出
    stderr=subprocess.DEVNULL
    )
        
        # 清理原始PDB文件（可选）
        os.remove(pdb_path)

# ========== 步骤3: 精炼处理 ==========
def woutpdb(infile, outfile):
    lines = open(infile).readlines()
    nums = [l.split()[5] for l in lines if l.startswith('ATOM')]
    nums = sorted(set(nums), key=int)
    
    with open(outfile, 'w') as fw:
        for line in lines:
            if line.startswith('ATOM'):
                parts = line.split()
                atom_num = parts[5]
                
                # 处理末端标记
                line_list = list(line)
                if atom_num == nums[0]:
                    line_list[18:20] = line_list[19:20] + ['5']
                elif atom_num == nums[-1]:
                    line_list[18:20] = line_list[19:20] + ['3']
                
                # 过滤原子
                if not (("P" in parts[2] and parts[5] == "1") or ("H" in parts[2])):
                    fw.write(''.join(line_list))

def woutpdb2(infile, outfile):
    with open(outfile, 'w') as fw:
        for line in open(infile):
            if line.startswith('ATOM') and 'H' not in line:
                fw.write(line)

def opt(inpdb, outpdb, steps):
    cuda_platform = Platform.getPlatformByName('CUDA')
    # 设置指定GPU设备（设备号从0开始）
    cuda_platform.setPropertyDefaultValue('CudaDeviceIndex', str(7))
    pdb = PDBFile(inpdb)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    
    # 建模和溶剂化
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)
    modeller.addSolvent(forcefield, padding=1*nanometer)
    
    # 创建模拟系统
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds
    )
    
    # 设置积分器
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    # simulation = Simulation(modeller.topology, system, integrator)
    simulation = Simulation(modeller.topology, system, integrator, cuda_platform)  # 添加平台参数
    simulation.context.setPositions(modeller.positions)
    
    # 能量最小化
    simulation.minimizeEnergy(maxIterations=steps)
    
    # 输出结果
    state = simulation.context.getState(getPositions=True)
    PDBFile.writeFile(simulation.topology, state.getPositions(), open(outpdb, 'w'))

def process_refine(dir):
    allowed_core_count = 64
    try:
        orig_affinity = os.sched_getaffinity(0)  # 获取原始亲和性
        total_cpus = os.cpu_count()
        print("Total CPUs available:", total_cpus)
        
        # 使用psutil获取每个核心的使用率（间隔1秒），得到一个列表，每个元素对应一个核心的使用率
        usage = psutil.cpu_percent(interval=1, percpu=True)
        # 按使用率从低到高排序，选择前 allowed_core_count 个核心作为空闲核心
        indices_sorted = sorted(range(len(usage)), key=lambda i: usage[i])
        if allowed_core_count > len(usage):
            allowed_core_count = len(usage)
        allowed_cpus = set(indices_sorted[:allowed_core_count])
        print("Selected CPU cores based on low usage:", allowed_cpus,
              "with usage percentages:", {i: usage[i] for i in allowed_cpus})
        os.sched_setaffinity(0, allowed_cpus)
        print("CPU affinity set to:", allowed_cpus)
    except Exception as e:
        print("Could not set CPU affinity:", e)
    """执行精炼处理"""
    for infile in glob.glob(os.path.join(dir, '*prefull.pdb')):
        base_name = os.path.basename(infile).replace('prefull.pdb', '')
        outfile = os.path.join(dir, f"{base_name}prefullrefine.pdb")
        if os.path.exists(outfile):
            continue       
        print("processing"+outfile)
        # 临时文件路径
        tmp1 = f"{infile}.tmp1"
        tmp2 = f"{infile}.tmp2"
        
        try:
            # 执行精炼流程
            woutpdb(infile, tmp1)
            with open(infile) as f:
                steps = max(min(len(f.readlines())//20, 1000), 10)
            opt(tmp1, tmp2, steps)
            woutpdb2(tmp2, outfile)
        finally:
            # 清理临时文件
            for f in [tmp1, tmp2]:
                if os.path.exists(f):
                    os.remove(f)
    try:
        os.sched_setaffinity(0, orig_affinity)
        print("CPU affinity restored to:", orig_affinity)
    except Exception as e:
        print("Could not restore CPU affinity:", e)                    

# ========== 主流程 ==========nohup python predict_refine.py > predict_refine2.log 2>&1 &
if __name__ == "__main__":
    # 步骤1: 生成预测文件
    predict(pre_in_dir,pre_out_dir)
    
    # 步骤2: Arena处理
    process_arena(pre_out_dir)
    
    # 步骤3: 精炼处理
    process_refine(pre_out_dir)
    
    print("所有处理步骤已完成！")