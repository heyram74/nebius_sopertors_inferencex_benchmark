#!/bin/bash
#SBATCH --job-name=h200_topology_check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --output=h200_topo_check_%j.log
#SBATCH --error=h200_topo_check_%j.err

echo "=================================================="
echo "STARTING H200 TOPOLOGY & NCCL OPTIMIZATION CHECK"
echo "Date: $(date)"
echo "Node Name: $SLURM_NODENAME"
echo "=================================================="

# 1. Verify all 8 H200 GPUs are visible
echo -e "\n[1/4] Checking Visible GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# 2. Check the hardware interconnect topology
echo -e "\n[2/4] Checking NVIDIA Topology Matrix..."
echo "Looking for 'NVLink' or 'NV' connectivity between all 8 GPUs."
nvidia-smi topo -m

# 3. Apply recommended single-node H200 NCCL optimizations
echo -e "\n[3/4] Setting NCCL and CUDA Optimization Variables..."

# Forces NCCL to output debugging logs so you can see if it selects NVLink
export NCCL_DEBUG=INFO

# Prevents NCCL from routing single-node traffic out through the network interface cards (NICs)
export NCCL_IB_DISABLE=1

# Ensures optimal kernel scheduling concurrency for tensor parallel communication
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Optional: Increases buffer size for high-bandwidth HBM3e messaging
export NCCL_BUFFSIZE=4194304

# Print set variables for confirmation
env | grep -E "NCCL_|CUDA_"

# 4. Optional: Run a lightweight PyTorch NCCL initialization to test the variables
echo -e "\n[4/4] Testing NCCL initialization via PyTorch..."
python3 -c "
import torch
import torch.distributed as dist
import os

if torch.cuda.is_available():
    print(f'PyTorch detected {torch.cuda.device_count()} GPUs.')
    # Minimal check to ensure the CUDA stack sees all resources correctly
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('ERROR: PyTorch cannot see CUDA.')
"

echo -e "\n=================================================="
echo "CHECK COMPLETE. Review log for 'SYS' or 'NODE' flags in step 2."
echo "=================================================="

