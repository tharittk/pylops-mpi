from mpi4py import MPI
import numpy as np
from numpy.testing import assert_allclose


from pylops_mpi.DistributedArray import Partition, local_split
import pylops_mpi
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = MPI.Get_processor_name()

# Test that DistributedArray works properly when engine is cuda
ny, nx = 8, 4
global_shape = (ny, nx)

arr = pylops_mpi.DistributedArray(
    global_shape, MPI.COMM_WORLD, Partition.SCATTER, axis=1
)
arr[:] = np.arange(
    arr.local_shape[0] * arr.local_shape[1] * arr.rank,
    arr.local_shape[0] * arr.local_shape[1] * (arr.rank + 1),
).reshape(arr.local_shape)

arr_np = arr.asarray()
if rank == 0:
    print(f"{arr_np=}")
    arr_base = np.arange(0, ny * nx).reshape((nx, ny)).T
    print(f"{arr_base=}")
    assert_allclose(arr_base, arr_np)
