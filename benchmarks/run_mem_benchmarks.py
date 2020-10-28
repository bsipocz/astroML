import benchit
import itertools

import XDGMM_benchmark

#%load_ext memory_profiler

#for N, comp, m_iter in itertools.product([100,200, 1000],[5,10],[250,500]):#0,2000,5000,20000,100000,200000], [5,10,15,20], [500,1000]]:

#in_ = {N:[N, 10, 500] for N in [1000,2000,5000,10000,20000,100000,200000]}

#timing = benchit.timings([XDGMM_benchmark.compute_XD_results,], in_, multivar=True, input_name='number-of-points')

#with open('numpy_benchmarks.pickle', 'wb') as outfile:
#    pickle.dump(timing, outfile)
#
#    print(timing)
#timing.plot(logx=True, save='numpy_benchmarks.png')

#%memit XDGMM_benchmark.compute_XD_results(N, comp, m_iter, True)
XDGMM_benchmark.compute_XD_results(1000, 5, 500)
