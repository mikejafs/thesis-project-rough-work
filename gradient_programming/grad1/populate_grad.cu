//nvcc -shared -o populate_grad.so populate_grad.cu -Xcompiler -fPIC

#include <stdio.h>

extern "C"
{
    __global__ void populate_gradient_kernel(
        double *gains,
        double *pr,
        double *pi,
        double *qr,
        double *qi,
        double *P,
        double *noise,
        double *outr,
        double *outi,
        long *ant_1_inds,
        long *ant_2_inds,
        int n_ant,
        int n_bl
    ){
        int k = blockIdx.x*blockDim.x + threadIdx.x;
        // int k1 = blockIdx.y*blockDim.y + threadIdx.y;
        // int k2 = blockIdx.z*blockDim.z + threadIdx.z;

        if (k < n_bl){
            // printf("%d", k);
            int k1 = ant_1_inds[k];
            int k2 = ant_2_inds[k];
            // if (k1 < n_ant/2){
            //     if (k2 < n_ant/2){

            //         // Compute the product of complex gains.
            //         double G_kr = gains[2*k1]*gains[2*k2] + gains[2*k1+1]*gains[2*k2+1];
            //         double G_ki = gains[2*k1+1]*gains[2*k2] - gains[2*k1]*gains[2*k2+1];
        
            //         // Compute the prefactor for the trace contribution.
            //         double prefac = 2 * noise[2*k] * P[k] / (G_kr*G_kr + G_ki*G_ki);
    
            //         double chi_sq_r = 2*(p[2*k1*n_ant + 2*k2] * q[2*k1*n_ant + 2*k2] + p[(2*k1+1)*n_ant + 2*k2+1] * q[(2*k1+1)*n_ant + 2*k2+1]);
            //         double chi_sq_i = 2*(p[(2*k1+1)*n_ant + 2*k2+1]*q[2*k1*n_ant + 2*k2] - p[(2*k1+1)*n_ant + 2*k2+1]*q[(2*k1+1)*n_ant + 2*k2+1]);
                    
            //         out[2*k1*n_ant + 2*k2] = prefac*G_kr - chi_sq_r;
            //         out[(2*k1+1)*n_ant + 2*k2+1] = prefac*G_ki - chi_sq_i;
                
            //     }
            // }
                
    
            // Compute the product of complex gains.
            double G_kr = gains[2*k1]*gains[2*k2] + gains[2*k1+1]*gains[2*k2+1];
            double G_ki = gains[2*k1+1]*gains[2*k2] - gains[2*k1]*gains[2*k2+1];
    
            // Compute the prefactor for the trace contribution.
            double prefac = 2 * noise[2*k] * P[k] / (G_kr*G_kr + G_ki*G_ki);
    
            // double chi_sq_r = 2*(p[k1*n_ant+k2]*q[k1*n_ant+k2] + p[(k1+1)*n_ant + k2+1]*q[(k1+1)*n_ant + k2+1]);
            // double chi_sq_i = 2*(p[(k1+1)*n_ant + k2+1]*q[k1*n_ant + k2] - p[(k1+1)*n_ant + k2+1]*q[(k1+1)*n_ant + k2+1]);
            
            // out[k1*n_ant + k2] = prefac*G_kr - chi_sq_r;
            // out[(k1+1)*n_ant + k2+1] = prefac*G_ki - chi_sq_i;

            // double chi_sq_rr = 2*(p[2*k1*n_ant + 2*k2]*q[2*k1*n_ant + 2*k2] + p[(k1+1)*n_ant + k2+1]*q[(k1+1)*n_ant + k2+1]);
            // double chi_sq_ri = 2*(p[k1*n_ant+k2]*q[k1*n_ant+k2] + p[(k1+1)*n_ant + k2+1]*q[(k1+1)*n_ant + k2+1]);
            // double chi_sq_ir = 2*(p[k1*n_ant+k2]*q[k1*n_ant+k2] + p[(k1+1)*n_ant + k2+1]*q[(k1+1)*n_ant + k2+1]);
            // double chi_sq_ii = 2*(p[(k1+1)*n_ant + k2+1]*q[k1*n_ant + k2] - p[(k1+1)*n_ant + k2+1]*q[(k1+1)*n_ant + k2+1]);
            
            // out[2*k1*n_ant + 2*k2] = prefac*G_kr - chi_sq_r;
            // out[2*k1*n_ant + 2*k2+1] = prefac*G_ki - chi_sq_i;
            // out[2*(k1+1)*n_ant + 2*k2] = prefac*G_ki - chi_sq_i;
            // out[2*(k1+1)*n_ant + 2*k2+1] = prefac*G_ki - chi_sq_i;
            
            // out[(2*k1+1)*n_ant + 2*k2+1] = prefac*G_kr - chi_sq_r;
            // out[(k1+1)*n_ant + 2*k2+3] = prefac*G_ki - chi_sq_i;
            // out[(2*k1+1)*n_ant + 2*k2+1] = prefac*G_kr - chi_sq_r;
            // out[(2*k1+1)*n_ant + 2*k2+1] = prefac*G_ki - chi_sq_i;


            // double chi_sq_r = 2*(p[2*k1*2*n_ant + 2*k2] * q[2*k1*2*n_ant + 2*k2] + p[(2*k1+1)*2*n_ant + 2*k2+1] * q[(2*k1+1)*2*n_ant + 2*k2+1]);
            // double chi_sq_i = 2*(p[(2*k1+1)*2*n_ant + 2*k2+1]*q[2*k1*2*n_ant + 2*k2] - p[(2*k1+1)*2*n_ant + 2*k2+1]*q[(2*k1+1)*2*n_ant + 2*k2+1]);


            double chi_sq_r = 2*(pr[k]*qr[k] + pi[k]*qi[k]);
            double chi_sq_i = 2*(pi[k]*qr[k] - pr[k]*qi[k]);

            if (k1 != k2){
                outr[k1*n_ant + k2] = prefac*G_kr - chi_sq_r;
                outi[k1*n_ant + k2] = prefac*G_ki - chi_sq_i; 
            }
            // outr[k1*n_ant + k2] = prefac*G_kr - chi_sq_r;
            // outi[k1*n_ant + k2] = prefac*G_ki - chi_sq_i;
            
            // out[2*k1*n_ant + 2*k2] = prefac*G_kr - chi_sq_r;
            // out[(2*k1+1)*n_ant + 2*k2+1] = prefac*G_ki - chi_sq_i;
        }
    }

    void populate_gradient(
        double *gains,
        double *pr,
        double *pi,
        double *qr,
        double *qi,
        double *P,
        double *noise,
        double *outr,
        double *outi,
        long *ant_1_inds,
        long *ant_2_inds,
        int n_ant,
        int n_bl
    ){
        int NUM_THREADS = 256;
        int NUM_BLOCKS = (n_bl + NUM_THREADS - 1)/NUM_THREADS;

        populate_gradient_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(
            gains,
            pr,
            pi,
            qr,
            qi,
            P,
            noise,
            outr,
            outi,
            ant_1_inds,
            ant_2_inds,
            n_ant,
            n_bl
        );
    }
}