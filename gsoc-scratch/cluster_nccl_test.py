"""
* Single-node Multi-GPU
$ srun -N 1 -n $(N_GPUS) --gpus-per-node=$(N_GPUS) python ...

* Multi-node One-GPU each
$ srun -N $(N_NODES) -n $(N_NODES) --gpus-per-node=1 python ...

* Multi-node Multi-GPU
$ srun -N $(N_NODES) -n $(N_NODES) * $(N_GPUS) --gpus-per-node=$(N_GPUS) python ...
"""

from mpi4py import MPI
import cupy as cp
import cupy.cuda.nccl as nccl
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = MPI.Get_processor_name()

local_rank = int(
    os.environ.get("SLURM_LOCALID")
    or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    or rank % cp.cuda.runtime.getDeviceCount()
)
device_id = local_rank
cp.cuda.Device(device_id).use()

print(
    f"rank: {rank}, host: {host}, local_slurm_id {local_rank}, device_id:{device_id}/{cp.cuda.runtime.getDeviceCount()}"
)

# Test data
n_elements = 16
dtype = cp.float32
buf = cp.arange(n_elements, dtype=dtype) + rank  # distinct per rank

# Initialize NCCL
if rank == 0:
    nccl_id = nccl.get_unique_id()
else:
    nccl_id = None
nccl_id = comm.bcast(nccl_id, root=0)

nccl_comm = nccl.NcclCommunicator(size, nccl_id, rank)

# AllReduce
nccl_comm.allReduce(
    buf.data.ptr,
    buf.data.ptr,
    n_elements,
    nccl.NCCL_FLOAT32,
    nccl.NCCL_SUM,
    cp.cuda.Stream.null.ptr,
)

print(f"Rank {rank}: result after AllReduce = {buf}")
