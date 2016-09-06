#!/usr/bin/env python

from mpi4py import MPI
import socket
import lj_dask as lj
import warnings

warnings.filterwarnings('ignore')

comm = MPI.COMM_WORLD

atoms = lj.make_cluster(int(4e4), 300, seed=comm.rank)

start = MPI.Wtime()
energy = lj.lj(atoms, do_forces=False).compute()
end = MPI.Wtime()

print('I am {} on {}'.format(comm.rank, socket.gethostname()))
print(' energy: {:20.10f}'.format(energy))
print('   time: {:.2f} s'.format(end-start))
