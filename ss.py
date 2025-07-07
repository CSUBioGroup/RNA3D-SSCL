import os
import numpy as np
from pathlib import Path
import subprocess

##############please modify##################
ViennaRNAbin = 'ViennaRNAbin'


#############################################
# def ss_feature(fastafile,saveprefix,outfasta):
#     ss = Extraxt_ss(fastafile, saveprefix)
#     mutN(fastafile, saveprefix, outfasta)
#     return ss
def ss_feature(fastafile,saveprefix,outfasta):
    Extraxt_ss(fastafile, saveprefix)
    mutN(fastafile, saveprefix, outfasta)
    return

def ss2matrix(ssstr):
    L = len(ssstr)
    ssmatrix = np.zeros([L, L])
    slist = []
    for i in range(L):
        if ssstr[i] in ['(', '<', '[', '{']:
            slist.append(i)
        elif ssstr[i] in [')', '>', ']', '}']:
            j = slist[-1]
            ssmatrix[i, j] = 1
            ssmatrix[j, i] = 1
            slist.pop(-1)
        elif ssstr[i] not in ['.', '-']:
            print('unknown ss state', ssstr[i], i)
            assert False
    return ssmatrix


def ViennaRNA_runner(fastafile, saveprefix_):
    saveprefix = os.path.abspath(saveprefix_)
    path = Path(saveprefix)
    seq = open(fastafile).readlines()[1].strip()
    workingdir = path.parent.absolute()
    name = os.path.basename(saveprefix)
    newfastafile = saveprefix + '.tmp.fasta'
    lines = [f'>{name}_ViennaRNA_name', seq]
    wfile = open(newfastafile, 'w')
    wfile.write('\n'.join(lines))
    wfile.close()
    os.chdir(workingdir)
    cmd = f'{ViennaRNAbin} --outfile={name}.ViennaRNAss -p  --noPS {os.path.abspath(newfastafile)}'
    print(cmd)
    os.system(cmd)
    ss_str = open(f'{saveprefix}.ViennaRNAss').readlines()[2].strip().split()[0]
    print(ss_str)
    ss1 = ss2matrix(ss_str)
    L = len(seq)
    pslines = open(os.path.join(workingdir, f'{name}_ViennaRNA_name_dp.ps')).readlines()
    pslines = [aline.strip() for aline in pslines]
    pslines = [aline.strip() for aline in pslines if
               aline.endswith('ubox') and len(aline.split()) == 4 and aline[0].isdigit()]

    ss2 = np.zeros([L, L])
    for aline in pslines:
        words = aline.split()
        i, j, score = int(words[0]) - 1, int(words[1]) - 1, float(words[2])
        ss2[i, j] = score
        ss2[j, i] = score
    os.remove(os.path.join(workingdir, f'{name}_ViennaRNA_name_dp.ps'))
    os.remove(newfastafile)
    os.remove(f'{saveprefix}.ViennaRNAss')
    if L == ss1.shape[0]:
        return np.concatenate([ss1[..., None], ss2[..., None]], axis=-1)
    else:
        print('ViennaRNA lengths does not id', L, ss1.sape)
        assert False

def run_contrafold(seq_file, params_file, working_dir):
    """
    运行contrafold命令并获取输出
    """
    command = f"cd {working_dir} && ./src/contrafold predict {seq_file} --params {params_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Command failed with error: {result.stderr}")
    return result.stdout

def run_contrafold2(seq_file, params_file, working_dir, bps_file):
    """
    Run contrafold command to generate both secondary structure and base pairing probabilities.
    """
    command = f"cd {working_dir} && ./src/contrafold predict {seq_file} --params {params_file} --posteriors 0.00001 {bps_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Command failed with error: {result.stderr}")
    return result.stdout

def parse_output(output):
    """
    解析contrafold的输出，提取二级结构信息
    """
    lines = output.split('\n')
    structure = None
    for i in range(len(lines)):
        if lines[i].startswith('>structure'):
            structure = lines[i + 1]
            break
    if structure is None:
        raise Exception("Structure not found in the output")
    return structure


def parse_bps(bps_file,length):
    """
    Parse the base-pairing probabilities from the bps.txt file.
    """
    bps_matrix = np.zeros((length, length), dtype=float)
    
    with open(bps_file, 'r') as f:
        for line in f:
            tokens = line.split()
            base = int(tokens[0]) - 1  # Convert to 0-based indexing
            for j in range(len(tokens)-2):
                i= j+2
                token_i = tokens[i].split(":")
                pair_base = int(token_i[0]) - 1  # Convert to 0-based indexing
                probability = float(token_i[1])
                bps_matrix[base, pair_base] = probability
                bps_matrix[pair_base, base] = probability  # Symmetric matrix
    return bps_matrix

def parse_dot_bracket(dot_bracket):
    """
    解析dot-bracket表示法，返回每个碱基的配对位置
    """
    stack = []
    pairs = {}
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
    return pairs

def create_contact_map(seq_length, pairs):
    """
    根据碱基配对信息生成contact map
    """
    contact_map = np.zeros((seq_length, seq_length), dtype=int)
    for i, j in pairs.items():
        contact_map[i, j] = 1
        contact_map[j, i] = 1
    return contact_map

def mxfold2_runner(fastafile, params_file, working_dir):
    """
    处理指定目录下的所有.seq文件，生成contact map并保存为.npy文件
    """


    bps_file =  fastafile.replace('.fasta', 'bps.txt')
    print(f"Processing {fastafile}...")
            
            # 运行contrafold并获取输出
    output = run_contrafold(fastafile, params_file, working_dir)
    run_contrafold2(fastafile, params_file, working_dir, bps_file)


            
            # 解析输出，获取二级结构
    dot_bracket = parse_output(output)
            
            # 生成contact map
    seq_length = len(dot_bracket)
    pairs = parse_dot_bracket(dot_bracket)
    contact_map = create_contact_map(seq_length, pairs)
    bps_matrix = parse_bps(bps_file,seq_length)       

    ss1 = np.concatenate([contact_map[..., None], bps_matrix[..., None]], axis=-1)
    return ss1


def Extraxt_ss(fastafile, ss_file):


    if os.path.exists(ss_file):
        # mutN(fastafile, ss_file, fastafile)
        # If the file exists, load and return its content
        return
    print(fastafile)
    efold_params_file = 'efold_params_file'
    efold_working_dir = 'efold_working_dir'
    ss1 = mxfold2_runner(fastafile,efold_params_file,efold_working_dir) 
    ss2 = ViennaRNA_runner(fastafile, ss_file.replace(".npy","") + '_vie')
    ss = np.concatenate([ss1, ss2], axis=-1)
    np.save(ss_file, ss)
    # mutN(fastafile, ss_file, fastafile)
    return

def mutN(fastafile, saveprefix, outfasta):
    # A U; GC
    conj = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    seq = open(fastafile).readlines()[1].strip()
    seq = list(seq)
    ss = np.load(saveprefix)[:, :, 0]
    newseq = []
    for i in range(len(seq)):
        if seq[i] not in ['A', 'G', 'C', 'U']:
            contacts = ss[i]  # L
            if contacts.sum() < 0.5:  # nno ss
                newseq.append('U')
            else:
                contacts2 = list(contacts.astype(int))
                if 1 in contacts2:
                    j = contacts2.index(1)
                    if seq[j] in ['A', 'U', 'G', 'C']:
                        newseq.append(conj[seq[j]])
                    else:
                        newseq.append('U')
                else:
                    newseq.append('U')
        else:
            newseq.append(seq[i])
    wfile = open(outfasta, 'w')
    wfile.write('>test\n')
    wfile.write(''.join(newseq))
    wfile.close()