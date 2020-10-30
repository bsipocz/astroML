import dask.config
dask.config.set({'array.chunk-size':'4kB'})
from XDGMM_daskarray_benchmark import compute_XD_results
compute_XD_results(n_points=200)
