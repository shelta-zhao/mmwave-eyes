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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    # Define the task to be run
    def process_task(i):
        mmEyes_pcd = mmEyesPCD(args.data_root, "cpu")
        yaml_path = f"{args.output_path}/adc_split/adc_list_{i+1}"
        mmEyes_pcd.run(yaml_path, device, save=args.save, display=args.display)
    
    # Use concurrent.futures to run tasks
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        futures = {executor.submit(process_task, i): i for i in range(args.process_num)}
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Task {futures[future]} generated an exception: {exc}")
            else:
                print(f"Task {futures[future]} completed successfully.")

    # Clean up the tmp folder after all processes are complete
    shutil.rmtree(args.output_path)
    

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
        # multi_process(args)
    else:
        print("Invalid pipeline option. Please choose 1 for the traditional pipeline.")
        sys.exit(1) 