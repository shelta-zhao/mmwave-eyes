"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : This script converts radar data to Point Cloud Data (PCD) format.
"""

import os
import sys
import torch
import shutil
import subprocess
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utility.tool_box import parse_arguments, adc_list_generate, split_yaml
from pipeline.adc_to_pcd import adc_to_pcd
from pipeline.mmEyes_pcd import mmEyesPCD

def multi_process(process_num, output_path="tmp"):
    """
    Run multiple processes to process the radar data.

    Parameters:
        process_num (int): The number of processes to run.
    """

    # Generate the list of data
    os.makedirs(output_path, exist_ok=True)
    adc_list_generate(data_root, output_file="adc_list.yaml")
    split_yaml("adc_list", output_path, process_num)

    # Run the processes
    processes = []
    for i in range(process_num):
        p = subprocess.Popen(["python", "main.py", "--pipeline", "2", "--yaml_path", f"{output_path}/adc_list_{i+1}"])
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.wait()

    # Remove the temporary files
    shutil.rmtree(output_path)

    print("All processes have finished.")

if __name__ == "__main__":
    
    # Parse options in the command line
    args = parse_arguments()

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Perform the processing pipeline
    if args.pipeline == 1:
        # Traditional pipeline to generate PCD from raw radar data
        point_cloud_data = adc_to_pcd(args.yaml_path, device, save=args.save, display=args.display)
    elif args.pipeline == 2:
        # Perform the mmEyes-PCD pipeline
        mmEyes_pcd = mmEyesPCD(args.data_root, device)
        mmEyes_pcd.run(args.yaml_path, device, save=args.save, display=args.display)
    else:
        print("Invalid pipeline option. Please choose 1 for the traditional pipeline.")
        sys.exit(1) 