"""Test the NCCL Collective Communication in Distributed Array class
Designed to run with n processes and each process has 1 GPU
$ mpiexec -n 3 pytest test_allreuce.py --with-mpi
"""

import numpy as np
import cupy as cp
import cupy.cuda.nccl as nccl
import pytest
from numpy.testing import assert_allclose
from mpi4py import MPI
import os
from pylops_mpi import DistributedArray, Partition
from pylops_mpi.DistributedArray import local_split


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = MPI.Get_processor_name()  # for debugging in multi-node


def initialize_nccl_comm():
    local_rank = int(
        os.environ.get("SLURM_LOCALID")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or rank % cp.cuda.runtime.getDeviceCount()
    )
    device_id = local_rank
    cp.cuda.Device(device_id).use()

    if rank == 0:
        nccl_id = nccl.get_unique_id()
    else:
        nccl_id = None
    nccl_id = comm.bcast(nccl_id, root=0)

    nccl_comm = nccl.NcclCommunicator(size, nccl_id, rank)
    return nccl_comm


nccl_comm = initialize_nccl_comm()

par1 = {
    "x": np.random.randn(size * 3),
    "partition": Partition.SCATTER,
}
par2 = {
    "x": np.random.randn(512),
    "partition": Partition.BROADCAST,
}


# AllReduce
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [par1, par2])
def test_allreduce(par):
    arr_mpi = DistributedArray.to_dist(
        par["x"], base_comm=MPI.COMM_WORLD, partition=par["partition"]
    )
    """
    TODO(tharitt): implement this
    arr_nccl = DistributedArray.to_dist(
        par["x"], base_comm=nccl_comm, partition=par["partition"]
    )
    there should be a notion of axis ?
    """
    # arr.base_comm.Allreduce(MPI.IN_PLACE, arr.local_array)  # in-place
    # or can also do with the recv buf

    result_mpi = arr_mpi.base_comm.allreduce(arr_mpi.local_array)
    """
    Mock-up of NCCL comm 
    To provide the same functionality like MPI, we may to write allreduce from allReduce() in NCCL
    i.e. when recv buffer is None, and you output the copy of the result after reduce
    """
    send_buf = cp.asarray(arr_mpi.local_array, dtype=cp.float32)
    recv_buf = cp.zeros_like(send_buf)

    nccl_comm.allReduce(
        send_buf.data.ptr,
        recv_buf.data.ptr,
        arr_mpi.local_array.shape[0],
        nccl.NCCL_FLOAT32,
        nccl.NCCL_SUM,
        cp.cuda.Stream.null.ptr,
    )

    # if rank == 0:
    assert_allclose(recv_buf.get(), result_mpi, rtol=1e-6)


# in-place - Braodcast
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [par1, par2])
def test_bcast(par):
    arr_mpi = DistributedArray.to_dist(
        par["x"], base_comm=MPI.COMM_WORLD, partition=Partition.BROADCAST
    )
    """
    DistributedArary uses self.base_comm.bcast(value)

    arr_nccl = DistributedArray.to_dist(
        par["x"], base_comm=nccl_comm, partition=Partition.BROADCAST
    )
    """

    if rank == 0:
        buf = cp.asarray(arr_mpi.local_array, dtype=cp.float32)
        # buf = cp.ones(arr_mpi.local_array.shape[0], dtype=cp.float32)

    else:
        buf = cp.zeros_like(arr_mpi.local_array, dtype=cp.float32)
        # buf = cp.zeros(9, dtype=cp.float32)

    # cp.cuda.Device().synchronize()
    print(f"Before {rank=} : {buf=}, {buf.size=}")
    nccl_comm.broadcast(
        buf.data.ptr,
        buf.data.ptr,
        buf.size,
        nccl.NCCL_FLOAT32,
        0,
        cp.cuda.Stream.null.ptr,
    )
    # cp.cuda.Device().synchronize()
    print(f"After: {rank=} : {buf.get()}, {type(buf)}")
    assert_allclose(buf.get(), arr_mpi.local_array)


if __name__ == "__main__":
    test_bcast(par1)
