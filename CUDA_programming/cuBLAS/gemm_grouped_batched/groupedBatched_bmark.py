import cupy as cp
import numpy as np
from simulate_params import *
from groupedBatched_func import *
from zp_puregpu_funcs_py import *
from cupyx.profiler import benchmark

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print()

#Parameter set up
n_ant = 500
n_eig = 3
n_src = 1

cp.random.seed(10)
spms = SimCorrcalParams(n_ant, n_eig, n_src, precision='float32', xp=cp)
edges = spms.edges()

#simulated matrices with correct shapes
sim_data = spms.sim_data()
noise = sim_data[0]
diff = sim_data[1]

#zeropad diff and noise 
zp_noise, nb, lb = zeroPad(noise, edges, return_inv=True)
zp_diff, nb, lb = zeroPad(diff, edges, return_inv=False)

# print(zp_diff)
# print(zp_noise)

#need this if wanting to compare to zped stuff since the true matmul is diff.T@N^-1@diff
noise = 1/noise

#set up the temp mat just before the mat mul we are interested in
temp = noise[..., None] * diff
zp_temp = zp_noise[..., None] * zp_diff

def cupy_block_mul(diff, tmp):
    return cp.transpose(diff, [0, 2, 1]) @ tmp

temp2 = cupy_block_mul(zp_diff, zp_temp)


#running the batched grouped matmul
C_array = groupedBatchedMatmul(diff, temp, edges)
# print(C_array)

print(np.allclose(temp2, C_array))



times = (benchmark(cupy_block_mul, (zp_diff, zp_temp), n_repeat = 100))
# gpu_times = gpu_times.split()
# gpu_cpu_t = float(gpu_times[3])/1e6
# gpu_gpu_t = float(gpu_times[14])/1e6

gpu_t_s = times.gpu_times
cpu_t_s = times.cpu_times

avg_gpu_t = cp.mean(gpu_t_s)
avg_cpu_t = cp.mean(cpu_t_s)

# print(gpu_cpu_t, gpu_gpu_t)
print('cpu:', avg_cpu_t, ' gpu:', avg_gpu_t)

times = (benchmark(groupedBatchedMatmul, (diff, temp, edges), n_repeat = 100))
# gpu_times = gpu_times.split()
# gpu_cpu_t = float(gpu_times[3])/1e6
# gpu_gpu_t = float(gpu_times[14])/1e6

gpu_t_s = times.gpu_times
cpu_t_s = times.cpu_times

avg_gpu_t = cp.mean(gpu_t_s)
avg_cpu_t = cp.mean(cpu_t_s)

# print(gpu_cpu_t, gpu_gpu_t)
print('cpu:', avg_cpu_t, ' gpu:', avg_gpu_t)








#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print()
