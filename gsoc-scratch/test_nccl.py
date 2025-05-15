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
# allreduce: take send buf, return value
# Allreduce: take send buf, recv buf


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [par1, par2])
def test_allreduce(par):
    # MPI
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

    result_mpi = arr_mpi.base_comm.allreduce(arr_mpi.local_array)
    """
    Mock-up of NCCL comm 
    To provide the same functionality like MPI, we may to write allreduce from allReduce() in NCCL
    i.e. when recv buffer is None, and you output the copy of the result after reduce
    """

    # NCCL
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

    assert_allclose(recv_buf.get(), result_mpi, rtol=1e-6)


# Braodcast
# bcast: take buf and root, return value
# Bcast: take buf, not return ? inp-place ?
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

    TODO(tharitt): test broadast in the same manner as __setitem__
    set item in SCATTER case looks problematic. It is called from DistributedArray
    but it calls local_array[index]
    i.e. with
        arr_mpi = DistributedArray.to_dist(
        np.arange(9), base_comm=MPI.COMM_WORLD, partition=Partition.SCATTER
        )
    arr_mpi[4] = 5
    this will fail while intuitively it should set the 4-th index to 5
    previously is called in arr[:] = arr_2.reshape(local_shape) so it is fine...
    and if you call arr[2] = 1, it will set local_array[2] = 1 for every partition and
    it looks like a broadcast - but not at the intention of a user
    """
    if rank == 0:
        buf = cp.asarray(arr_mpi.local_array, dtype=cp.float32)
    else:
        buf = cp.zeros_like(arr_mpi.local_array, dtype=cp.float32)

    nccl_comm.broadcast(
        buf.data.ptr,
        buf.data.ptr,
        buf.size,
        nccl.NCCL_FLOAT32,
        0,
        cp.cuda.Stream.null.ptr,
    )

    assert_allclose(buf.get(), arr_mpi.local_array)


# Reduce
# reduce: take send buf, return value
# Reduce: take send buf, recv buf
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [par1, par2])
def test_reduce(par):
    arr_mpi = DistributedArray.to_dist(
        par["x"], base_comm=MPI.COMM_WORLD, partition=par["partition"]
    )

    # if rank == 0:
    #     arr_mpi.base_comm.Reduce(MPI.IN_PLACE, arr_mpi.local_array, MPI.SUM, root=0)
    # else:
    #     arr_mpi.base_comm.Reduce(arr_mpi.local_array, rank)

    # reduce to root (but not overwrite)
    result_mpi = arr_mpi.base_comm.reduce(arr_mpi.local_array)

    recv_buf = cp.zeros(arr_mpi.local_array.shape[0], dtype=cp.float32)
    send_buf = cp.asarray(arr_mpi.local_array, dtype=cp.float32)
    nccl_comm.reduce(
        send_buf.data.ptr,
        recv_buf.data.ptr,
        recv_buf.size,
        nccl.NCCL_FLOAT32,
        nccl.NCCL_SUM,
        0,
        cp.cuda.Stream.null.ptr,
    )

    if rank == 0:  # only root has the output buffer defined
        assert_allclose(result_mpi, recv_buf.get(), rtol=1e-6)


# AllGather
# allgather: take send buf, return list[]
# Allgather: take send buf, recv buf (nrank * send_buf.size)
# Allgatherv: ?
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("par", [par1, par2])
def test_allgather(par):
    # MPI
    arr_mpi = DistributedArray.to_dist(
        par["x"], base_comm=MPI.COMM_WORLD, partition=par["partition"]
    )

    result_mpi = np.concatenate(arr_mpi.base_comm.allgather(arr_mpi.local_array))

    # NCCL
    send_buf = cp.asarray(arr_mpi.local_array, dtype=cp.float32)
    recv_buf = cp.zeros(size * arr_mpi.local_array.shape[0], dtype=cp.float32)
    nccl_comm.allGather(
        send_buf.data.ptr,
        recv_buf.data.ptr,
        send_buf.size,
        nccl.NCCL_FLOAT32,
        cp.cuda.Stream.null.ptr,
    )

    assert_allclose(result_mpi, recv_buf.get())


# Observation: NCCL API requires the buffer and does not return value like MPI
# You need to wrap it yourself

if __name__ == "__main__":
    test_allgather(par1)
