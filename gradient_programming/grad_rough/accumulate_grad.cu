//nvcc -shared -o accumulate_grad.so accumulate_grad.cu -Xcompiler -fPIC

#include <stdio.h>

extern "C"
{
    __global__ void accumulate_gradient_kernel(
        double *gains,
        double *s,
        double *t,
        double *P,
        double *noise,
        double *out,
        long *ant_1_inds,
        long *ant_2_inds,
        int n_bl
    ){
        int k = blockIdx.x*blockDim.x + threadIdx.x;
        if (k < n_bl){
            int k1 = ant_1_inds[k];
            int k2 = ant_2_inds[k];
    
            // Accumulate the contribution from the chi-squared gradient.
            out[2*k1] -= 2 * (gains[2*k2]*s[k] - gains[2*k2+1]*t[k]);
            out[2*k2] -= 2 * (gains[2*k1]*s[k] + gains[2*k1+1]*t[k]);
            out[2*k1+1] -= 2 * (gains[2*k2+1]*s[k] + gains[2*k2]*t[k]);
            out[2*k2+1] -= 2 * (gains[2*k1+1]*s[k] - gains[2*k1]*t[k]);

            // Try with atomic add to sidestep memory issues -> Doesn't seem to work...
            // atomicAdd(&out[2*k1], -2 * (gains[2*k2]*s[k] - gains[2*k2+1]*t[k]));
            // atomicAdd(&out[2*k2], -2 * (gains[2*k1]*s[k] + gains[2*k1+1]*t[k]));
            // atomicAdd(&out[2*k1+1], -2 * (gains[2*k2]*s[k] + gains[2*k2]*t[k]));
            // atomicAdd(&out[2*k2+1], -2 * (gains[2*k1]*s[k] - gains[2*k1]*t[k]));

    
            // Compute the product of complex gains.
            double G_kr = gains[2*k1]*gains[2*k2] + gains[2*k1+1]*gains[2*k2+1];
            double G_ki = gains[2*k1+1]*gains[2*k2] - gains[2*k1]*gains[2*k2+1];
    
            // Compute the prefactor for the trace contribution.
            double prefac = 2 * noise[2*k] * P[k] / (G_kr*G_kr + G_ki*G_ki);
    
            // Now accumulate contributions from the trace.
            out[2*k1] += prefac * (G_kr*gains[2*k2] - G_ki*gains[2*k2+1]);
            out[2*k2] += prefac * (G_kr*gains[2*k1] + G_ki*gains[2*k1+1]);
            out[2*k1+1] += prefac * (G_kr*gains[2*k2+1] + G_ki*gains[2*k2]);
            out[2*k2+1] += prefac * (G_kr*gains[2*k1+1] - G_ki*gains[2*k1]);
        
            // atomicAdd(&out[2*k1], prefac*(G_kr*gains[2*k2] - G_ki*gains[2*k2+1]));
            // atomicAdd(&out[2*k2], prefac*(G_kr*gains[2*k1] - G_ki*gains[2*k1+1]));
            // atomicAdd(&out[2*k1+1], prefac*(G_kr*gains[2*k2+1] - G_ki*gains[2*k2]));
            // atomicAdd(&out[2*k2+1], prefac*(G_kr*gains[2*k1+1] - G_ki*gains[2*k1]));
        }
    }

    void accumulate_gradient(
        double *gains,
        double *s,
        double *t,
        double *P,
        double *noise,
        double *out,
        long *ant_1_inds,
        long *ant_2_inds,
        int n_bl
    ){
        int NUM_THREADS = 256;
        int NUM_BLOCKS = (n_bl + NUM_THREADS - 1)/NUM_THREADS;

        accumulate_gradient_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(
            gains,
            s,
            t,
            P,
            noise,
            out,
            ant_1_inds,
            ant_2_inds,
            n_bl
        );
    }
}