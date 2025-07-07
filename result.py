import os
import re
import subprocess
import argparse

# ========== 配置参数 ==========
#Please configure the corresponding parameters
real_dir = ""
results = []
suffix = "preprefull"
pred_dir = ""
tmscore = 1
output_file = ''


def run_rnaalign(real_pdb, pred_pdb, tmscore_option):
    cmd = f"./RNAalign {real_pdb} {pred_pdb} -TMscore {tmscore_option}"
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running RNAalign: {e}")
        return None

def parse_output(output):
    data = {'tm': None, 'rmsd': None, 'length': None}
    
    # 更宽松的解析逻辑
    try:
        # 解析长度（允许任意空格和大小写）
        length_match = re.search(
            r'Length\s*of\s*Chain_1\s*:\s*(\d+)', 
            output, 
            re.IGNORECASE
        )
        if length_match:
            data['length'] = int(length_match.group(1).strip())

        # 解析RMSD（兼容不同分隔符格式）
        rmsd_match = re.search(
            r'RMSD\s*[=:]\s*([\d.]+)',  # 匹配 "RMSD=1.20" 或 "RMSD: 1.20"
            output
        )
        if rmsd_match:
            data['rmsd'] = float(rmsd_match.group(1).strip())

        # 解析第一个TM-score（兼容科学计数法）
        tm_matches = re.findall(
            r'TM-score\s*=\s*([\d.]+(?:e-?\d+)?)',  # 匹配 0.81266 或 1.2e-5
            output,
            re.IGNORECASE
        )
        if tm_matches:
            data['tm'] = float(tm_matches[0])  # 取第一个匹配值

    except (ValueError, IndexError) as e:
        print(f"Parsing error: {str(e)}")
    if data['tm'] is None or data['rmsd'] is None:
        print(output)
        print(data)
    
    return data

def calculate_stats(results):
    intervals = [
        (0, 20, []),
        (20, 50, []),
        (50, 100, []),
        (100, 200, []),
        (200, float('inf'), []),
    ]
    all_data = []

    for res in results:
        if None in res.values():
            continue
        all_data.append((res['tm'], res['rmsd']))
        length = res['length']
        for i, (start, end, data) in enumerate(intervals):
            if start <= length < end:
                data.append((res['tm'], res['rmsd']))
                break
        else:
            intervals[-1][2].append((res['tm'], res['rmsd']))

    stats = []
    for start, end, data in intervals:
        if not data:
            stats.append((f"{start}-{end}" if end != float('inf') else f">{start}", 0, 0, 0))
            continue
        
        avg_tm = sum(tm for tm, _ in data) / len(data)
        avg_rmsd = sum(rmsd for _, rmsd in data) / len(data)
        range_name = f"{start}-{end}" if end != float('inf') else f">{start}"
        stats.append((range_name, len(data), avg_tm, avg_rmsd))
    
    overall_avg = (0, 0)
    if all_data:
        overall_avg = (
            sum(tm for tm, _ in all_data) / len(all_data),
            sum(rmsd for _, rmsd in all_data) / len(all_data)
        )
    
    return stats, overall_avg

def parse_args():
    parser = argparse.ArgumentParser(description='Compare RNA structures using RNAalign')
    # 默认参数设置（适配IDE直接运行）
    parser.add_argument('real_dir', default="real_dir", help='Directory containing real PDB structures')
    parser.add_argument('pred_dir', default="pred_dir", help='Directory containing predicted PDB structures')
    parser.add_argument('output_file', default="results.txt", help='Output TXT file path')
    parser.add_argument('--suffix', default="preprefull", help='Suffix added to predicted filenames')
    parser.add_argument('--tmscore', type=int, default=1, help='TM-score option (default: 1)')
    return parser.parse_args()


def compute_tm(pred_dir, real_dir , suffix = "preprefull"):
    real_dir 
    results = []
    suffix = "preprefull"
    tmscore = 1

    for real_file in os.listdir(real_dir):
        if not real_file.endswith('.pdb'):
            continue
        
        base_name = os.path.splitext(real_file)[0]
        pred_file = f"{base_name}{suffix}.pdb"
        
        real_path = os.path.abspath(os.path.join(real_dir, real_file))
        pred_path = os.path.abspath(os.path.join(pred_dir, pred_file))
        
        if not os.path.exists(pred_path):
            continue
        
        output = run_rnaalign(real_path, pred_path, tmscore)
        if not output:
            continue
        
        data = parse_output(output)
        if None in data.values():
            print(output)
            print(f"Warning: Failed to parse output for {real_file}")
            continue
        results.append({
    'name': base_name,
    'tm': data['tm'],
    'rmsd': data['rmsd'],
    'length': data['length']
})
        
    stats, (overall_tm, overall_rmsd) = calculate_stats(results)       
    for range_name, count, avg_tm, avg_rmsd in stats:
        print(f"{range_name}\t{count}\t{avg_tm:.4f}\t{avg_rmsd:.2f}\n")        
    print(f"\nOverall Average\t{len(results)}\t{overall_tm:.4f}\t{overall_rmsd:.2f}\n")



def main():


    for real_file in os.listdir(real_dir):
        if not real_file.endswith('.pdb'):
            continue
        
        base_name = os.path.splitext(real_file)[0]
        pred_file = f"{base_name}{suffix}.pdb"
        
        real_path = os.path.abspath(os.path.join(real_dir, real_file))
        pred_path = os.path.abspath(os.path.join(pred_dir, pred_file))
        
        if not os.path.exists(pred_path):
            print(f"Warning: Predicted file {pred_path} not found. Skipping.")
            continue
        
        output = run_rnaalign(real_path, pred_path, tmscore)
        if not output:
            continue
        
        data = parse_output(output)
        if None in data.values():
            print(output)
            print(f"Warning: Failed to parse output for {real_file}")
            continue
        results.append({
    'name': base_name,
    'tm': data['tm'],
    'rmsd': data['rmsd'],
    'length': data['length']
})
        


    # Write individual results
    with open(output_file, 'w') as f:
        # Write header
        f.write("RNA\tTM-score\tRMSD\tLength\n")
        
        # Write individual results
        for res in results:
            f.write(f"{res['name']}\t{res['tm']:.4f}\t{res['rmsd']:.2f}\t{res['length']}\n")
        
        # Calculate and write statistics
        stats, (overall_tm, overall_rmsd) = calculate_stats(results)
        
        f.write("\nLength Range\tCount\tAvg TM-score\tAvg RMSD\n")
        for range_name, count, avg_tm, avg_rmsd in stats:
            f.write(f"{range_name}\t{count}\t{avg_tm:.4f}\t{avg_rmsd:.2f}\n")
        
        f.write(f"\nOverall Average\t{len(results)}\t{overall_tm:.4f}\t{overall_rmsd:.2f}\n")

if __name__ == '__main__':
    main()



