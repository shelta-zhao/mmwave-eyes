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
from pathos.multiprocessing import ProcessingPool as Pool
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utility.tool_box import parse_arguments, adc_list_generate, split_yaml
from pipeline.adc_to_pcd import adc_to_pcd
from pipeline.mmEyes_pcd import mmEyesPCD

def multi_process(args):
    """
    Run multiple processes to process the radar data.

    Parameters:
        args (Namespace): The arguments from the command line
    """

    # Generate the list of data
    os.makedirs(args.output_path, exist_ok=True)
    adc_list_generate(args.data_root, output_file=f"{args.output_path}/adc_list")
    split_yaml(f"{args.output_path}/adc_list", args.output_path, args.process_num)

    # Create mmEyesPCD instance for each process and run it in parallel
    mmEyes_pcd = mmEyesPCD(args.data_root, device)
    def process_task(i):
        yaml_path = f"{args.output_path}/adc_split/adc_list_{i+1}"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mmEyes_pcd.run(yaml_path, device, save=args.save, display=args.display)
        
    # Use multiprocessing to run tasks
    with Pool(processes=args.process_num) as pool:
        pool.map(process_task, range(args.process_num))

    # Clean up the tmp folder after all processes are complete
    # shutil.rmtree(args.output_path)

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
        multi_process(args)
    else:
        print("Invalid pipeline option. Please choose 1 for the traditional pipeline.")
        sys.exit(1) 