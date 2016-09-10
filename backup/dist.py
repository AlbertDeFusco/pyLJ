from dask.distributed import Executor
import dask
import os
import lj_dask as lj

NCPUS = os.cpu_count()

def do_compute(seed, size=int(4e4), radius=300):
    with dask.set_options(get=dask.threaded.get):
        cluster = lj.make_cluster(size, radius, seed)
        energy = lj.lj(cluster, do_forces=False)

        return energy.compute(num_workers=NCPUS)


e = Executor('127.0.0.1:8786')
e.restart()
e.upload_file('lj_dask.py')

futures = e.map(do_compute, range(2))
out = e.gather(futures)
print(out)
