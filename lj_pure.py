import numpy as np
import sys

def make_cluster(natoms, radius=20, seed=1981):
    np.random.seed(seed)
    cluster = ((np.random.normal(0, radius, (natoms,3))-0.5)).tolist()
    return cluster

def lj(r):
    sr6 = (1./r)**6
    pot = 4.*(sr6*sr6 - sr6)
    return pot

def distance(atom1, atom2):
    dx = atom2[0] - atom1[0]
    dy = atom2[1] - atom1[1]
    dz = atom2[2] - atom1[2]

    r = (dx*dx + dy*dy + dz*dz)**0.5

    return r

def potential(cluster):
    energy = 0.0
    for i in range(len(cluster)-1):
        for j in range(i+1,len(cluster)):
            r = distance(cluster[i],cluster[j])
            e = lj(r)
            energy += e

    return energy

def main(natoms=100):
    print('A {:d} atom cluster'.format(natoms))
    atoms = make_cluster(natoms)
    energy = potential(atoms)
    print('  Total energy: {:.4f}'.format(energy))

if __name__ == '__main__':
    try:
        main(int(sys.argv[1]))
    except IndexError:
        main()
