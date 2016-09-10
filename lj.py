import sys
import numpy as np
import dask.array as da
import scipy.optimize as opt
import os
import warnings

warnings.filterwarnings('ignore')

NCPUS = os.cpu_count()
SIGMA = 3.40
EPSILON = 0.238

def make_cluster(natoms, radius=2, seed=1981):
    np.random.seed(seed)
    cluster = np.random.normal(0, radius, size=(natoms,3))-0.5
    return cluster


def potential(r2):
    sr6 = (SIGMA/r2)**3
    return 4.*EPSILON*(sr6**2 - sr6)


def gradient(r2):
    sr2 = (SIGMA/r2)
    return 24.*EPSILON*(2.*sr2**7 - sr2**4)


def distance_matrix(cluster):
    diff = cluster[:, np.newaxis, :] - cluster[np.newaxis, :, :]
    r2 = (diff**2).sum(-1)
    r = np.sqrt(r2)
    return diff, r2, r


def evaluate(cluster, do_forces=True):
    if cluster.ndim == 1:
        cluster = cluster.reshape(-1,3)

    if NCPUS > cluster.shape[0]:
        chunks=1
    else:
        chunks = cluster.shape[0]//NCPUS

    darr = da.from_array(cluster, chunks=chunks)

    diff, r2, _ = distance_matrix(darr)

    energy = da.nansum(potential(r2))/2.

    if do_forces:
        forces = da.nansum(gradient(r2)[:,:,np.newaxis]*diff,axis=0)
        return energy.compute(), forces.compute()
    else:
        return energy.compute()



def write(cluster, filename='optimized.xyz', append=False, gradient=None):
    mode = 'ab' if append else 'wb'

    if gradient is None:
        fmt = 'Ar %12.8f %12.8f %12.8f'
        c = cluster
    else:
        fmt = 'Ar %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f'
        c = np.concatenate([cluster,gradient], axis=1)

    with open(filename, mode) as f:
        np.savetxt(f, c, header='{:d}\n'.format(cluster.shape[0]),
                   comments='', fmt=fmt)

def optimize(cluster):
    def f(arr):
        e,g = evaluate(arr, do_forces=True)
        return e, g.flatten()

    ret = opt.minimize(f, cluster, jac=True,
                       method='L-BFGS-B', options={'disp':True})
    return ret.x.reshape(-1,3)

def main(natoms=3):
    print('A {:d} atom cluster'.format(natoms))
    atoms = make_cluster(natoms)
    out = optimize(atoms)
    _, _, dmat = distance_matrix(out)
    print('Distances (angstrom)')
    print(dmat)

if __name__ == '__main__':
    try:
        main(int(sys.argv[1]))
    except IndexError:
        main()
