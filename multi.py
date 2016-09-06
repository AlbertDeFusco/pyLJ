#!/usr/bin/env python

from mpi4py import MPI
import socket
import lj
import warnings

warnings.filterwarnings('ignore')

comm = MPI.COMM_WORLD

atoms = lj.make_cluster(int(1e4), 8000)

start = MPI.Wtime()
energy = lj.lj(atoms, do_forces=False)
end = MPI.Wtime()

print('I am {} on {}'.format(comm.rank, socket.gethostname()))
print(' energy: {:.2f}'.format(energy))
print('   time: {:.2f} s'.format(end-start))
