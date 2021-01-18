#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <deque>
#include <array>
#include <time.h>
#include <chrono>
#include <thrust/copy.h>
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
#include "device_functions.h"

constexpr int N = 50;
constexpr int RATE = 80;

// flag =0 solo sequenziale, flag =1 entrambi, flag >1 solo parallelo
constexpr int flag = 1;


__constant__ int a[N];
__device__ int d_count;
__device__ int d_n[N*N];



cudaError_t clique_launcher(std::vector<int>& degrees, std::vector<int>& neighbours, std::vector<int>& indexes);

void seq_intersection(std::deque<int>& intersection, std::deque<int>& local_neight, std::deque<int> &inters_local) {
    int i = 0, j = 0, count = 0;
    std::deque<int> tmp(std::min(intersection.size(), local_neight.size()));
    while (i < intersection.size() && j < local_neight.size()) {
        if (intersection[i] < local_neight[j])
            i++;
        else if (local_neight[j] < intersection[i])
            j++;
        else
        {
            tmp[count] = local_neight[j];
            j++;
            i++;
            count++;
        }
        inters_local = std::deque<int>(tmp.begin(), tmp.begin() + count);
    }
}

void rec_seq_clique(const std::vector<int> &degrees, const std::vector<int>&neighbours, std::deque<int> &intersection,const std::vector<int>&indexes, std::array<int, N> &max_clique_seq, std::array<int,N> &tmp, int &best_size, int tmp_index, int current) {
    int crt;
    std::deque<int> local_neight(degrees[current]);
   
    
    for (int i = 0; i < degrees[current]; i++) {
        local_neight[i] = neighbours[indexes[current] + i];
    }
    std::sort(local_neight.begin(), local_neight.end());
    std::deque<int> inters_local;
    if (tmp_index > 1) {
        seq_intersection(intersection, local_neight, inters_local);
    }
    else{
        inters_local = local_neight;
    }
    
    while (!inters_local.empty() && inters_local.size()+tmp_index>best_size) {
        
        crt = inters_local[0];
        inters_local.pop_front();
        tmp[tmp_index] = crt;
        rec_seq_clique(degrees, neighbours, inters_local, indexes, max_clique_seq, tmp, best_size, tmp_index+1, crt);
    }
    
    if (tmp_index > best_size) {
        best_size = tmp_index;
        for (int i = 0; i < N; i++) {
            if(i<=tmp_index)
                max_clique_seq[i] = tmp[i];
            else
                max_clique_seq[i] = -1;
        }
    }
}

//codice sequenziale
void seq_clique(const std::vector<int> &degrees, const std::vector<int>&neighbours, const std::vector<int> &indexes, std::array<int, N> &max_clique_seq, std::array<int,N> &tmp, int &best_size) {
    best_size = 0;
    for (int i = 0; i < N; i++) {
        tmp[0] = i;
        std::deque<int> d = std::deque<int>(0);
        rec_seq_clique(degrees, neighbours, d, indexes, max_clique_seq, tmp, best_size, 1, i);
        
    }
}

__host__ std::vector<int> create_graph(std::vector<int>& degrees, int MaxNeightbours) {
    std::vector<int> temp_neighbours(N * MaxNeightbours);

    int sizeOfN = 0;
    int r = 0;
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {

            r = rand() % 100;
            if (r < RATE) {
                if (degrees[i] >= MaxNeightbours || degrees[j] >= MaxNeightbours) printf("Troppi vicini, balzo\n");
                else {
                    //printf("%d, %d, %d, %d\n", i, j, (i * MaxNeightbours) + degrees[i], (j * MaxNeightbours) + degrees[j]);
                    temp_neighbours[i * MaxNeightbours + degrees[i]] = j;
                    temp_neighbours[j * MaxNeightbours + degrees[j]] = i;
                    degrees[i]++;
                    degrees[j]++;
                    sizeOfN += 2;
                }
            }
        }
    }
    std::vector <int> to_ret(sizeOfN);
    int counter = 0;
    int count = 0, current = 0, currentNode = 0;
    for (int i = 0; i < N * MaxNeightbours; i++) {
        if (i != 0 && i % MaxNeightbours == 0) {
            count = 0;
            currentNode++;
        }
        if (count < degrees[currentNode]) {
            to_ret[current] = temp_neighbours[i];
            counter++;
            count++;
            current++;
        }
    }
    return to_ret;
}

int main() {
    
    
    
    std::array<int,N> tmp_max_clique_seq;

    thrust::device_vector<int> dev_max_clique(N, -1);
    //sizes per le malloc
    constexpr int MaxNeightbours = N;
    srand((unsigned)time(NULL));
    //printf("max vicini %d\n", MaxNeightbours);

    

    //serve eprchè non so la dimensione, quindi ne faccio una amsisma poi accorcio
    //temp_neightbours è usato con dimensione NxN
    

    std::vector<int> degrees(N);

    int best_size = 0;

    std::vector<int> neighbours= create_graph(degrees, MaxNeightbours);
    

    std::vector<int> indexes(N);
    indexes[0] = 0;
    int sum = degrees[0];
    //sistemazione dell'array degli indici
    for (int i = 1; i < N; i++) {
        
        indexes[i] = sum;
        sum += degrees[i];
        
    }

    //la seconda è la clique path in corso nella ricorsione
    std::array<int, N> max_clique_seq;

    for (int i = 0; i < N; i++) {
        max_clique_seq[i] = -1;
    }

   

    for (int i = 0; i < N; i++) {
        tmp_max_clique_seq[i] = -1;
    }

    /*int f = 0;
    int count = 0;
    for (int i = 0; i < indexes.size(); i++) {
        for (int j = 0; j < degrees[i]; j++) {
            std::cout << neighbours[indexes[i]+j] << ' ';
        }
        std::cout << "  //  ";
    }
    std::cout << '\n';*/

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    if (flag <= 1) {
        printf("\nInizio il sequenziale:\n");
        seq_clique(degrees, neighbours, indexes, max_clique_seq, tmp_max_clique_seq, best_size);
        printf("\n clique seq lunga %d: \n", best_size);
        for (int i = 0; i < best_size; i++) {
            printf("--> %d", max_clique_seq[i]);
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<float, std::milli> ms = end - begin;
    std::cout << "\n\nTime difference = " << ms.count() << "[ms] " << ms.count() / 1000 << "[s] " << ms.count() / 60000 << "[m] " << std::endl;

    
    //necessario epr fare memcpy su gpu
    for (int i = 0; i < N; i++) {
        tmp_max_clique_seq[i] = -1;
    }
    
    
    if (flag >= 1) {
        printf("\nInizio il parallelo:\n");
        cudaError_t cudaStatus = clique_launcher(degrees, neighbours, indexes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "kernel fail!");
            return 1;
        }
    }    
}



__global__ void parall_intersection(int n, int m,int start, int* inters_local) {
    int idx = blockIdx.x;
    int idy = threadIdx.x;
    if (idx < n && idy < m) {
        int inter = a[idx];
        int lcl_n= d_n[start+idy];
        if (inter == lcl_n) {
            int i = atomicAdd(&d_count, 1);
            inters_local[i] = inter;
        }
    }
    
}

cudaError_t rec_par_clique(std::vector<int>& degrees, std::vector<int>& neighbours, int int_start, std::vector<int>& intersection, std::vector<int>& indexes, std::array<int, N>& tmp, int& best_size, int tmp_index, int current, int* to_ret) {
    cudaError_t cudaStatus = cudaSuccess;
    std::vector<int> intersection_local;
    int count = 0;
    int sz = (int)intersection.size() - int_start;
    int min = std::min(sz, degrees[current]);
    if (tmp_index > 1) {
        int* dev_c;
        cudaStatus = cudaMemcpyToSymbol(a, &intersection[int_start], sz * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemCpyToSymbol failed! %s \n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
        cudaStatus = cudaMemcpyToSymbol(d_count, &count, sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemCpyToSymbol2 failed! %s \n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
        cudaStatus = cudaMalloc((void**)&dev_c, min * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Cudamalloc failed!");
            return cudaStatus;
        }
        parall_intersection <<<sz, degrees[current] >>> (sz,
            degrees[current],
            indexes[current],
            dev_c);
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\ncudaDeviceSynchronize returned error code %d after launching Kernel!  %s \n", cudaStatus, cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaStatus = cudaMemcpyFromSymbol(&count, d_count, sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemCpyFROMSymbol failed! %s \n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
        if (count > 0) {
            intersection_local = std::vector<int>(count);

            cudaStatus = cudaMemcpy(&intersection_local[0], dev_c, count * sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemCpyFROMSymbol2 failed!\n %s", cudaGetErrorString(cudaStatus));
                return cudaStatus;
            }
        }
        cudaFree(dev_c);
    }
    else {
        intersection_local = std::vector<int>(neighbours.begin() + indexes[current], neighbours.begin() + indexes[current] + degrees[current]);
        count = intersection_local.size();
    }
    int i=0;
    while (i<count-1  && count + tmp_index - i > best_size) {
        
        tmp[tmp_index] = intersection_local[i];
        cudaStatus = rec_par_clique(degrees, neighbours, i+1, intersection_local, indexes, tmp, best_size, tmp_index + 1, intersection_local[i], to_ret);
        if (cudaStatus != cudaSuccess) {
            return cudaStatus;
        }
        i++;
    }

    if (tmp_index >= best_size) {
        
        if (count > 0) {
            tmp[tmp_index] = intersection_local[i];
            tmp_index++;
            best_size = tmp_index;
        }
        else {
            best_size = tmp_index;
        }
            
        cudaStream_t s;
        cudaStatus = cudaStreamCreate(&s);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSteamCreate failed!\n");
            return cudaStatus;
        }

        cudaMemcpyAsync(to_ret, &tmp, tmp_index * sizeof(int), cudaMemcpyHostToDevice, s);
    }
    return cudaStatus;
}

cudaError_t parallel_clique(std::vector<int>& degrees, std::vector<int>& neighbours, std::vector<int>& indexes, std::array<int, N>& tmp, int& best_size, int*  to_ret) {
    best_size = 0;
    cudaError_t cudaStatus = cudaSuccess;
    for (int i = 0; i < N; i++) {
        tmp[0] = i;
        std::vector<int>  d = std::vector<int>(0);
        cudaStatus=rec_par_clique(degrees, neighbours, 0, d, indexes, tmp, best_size, 1, i, to_ret);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "recursive %d failed\n", i);
            return cudaStatus;
        }

    }
    return cudaStatus;
    
}


//qui eseguirò il codice parallelo
cudaError_t clique_launcher(std::vector<int>& degrees, std::vector<int>& neighbours, std::vector<int>& indexes)
{
    cudaError_t cudaStatus = cudaSuccess;


    int d_best_size = 0;
    std::array<int, N> tmp;
    for (int i = 0; i < N; i++) {
        tmp[i] = -1;
    }
    int* dev_max_clique, *max_clique;
    int sz = (int)neighbours.size();
    max_clique=(int*)malloc(N * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_max_clique, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpyToSymbol(d_n, &neighbours[0], sz * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaNeightMemCpyToSymbol failed! %s \n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    cudaStatus = parallel_clique(degrees, neighbours, indexes, tmp, d_best_size, dev_max_clique);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda failed :(( failed! %s", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<float, std::milli> ms = end - begin;
    
    printf("\n clique rec lunga %d: \n", d_best_size);

    cudaStatus = cudaMemcpy(max_clique, dev_max_clique, d_best_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda memcpyfromsymbol maxclique failed\n");
        return cudaStatus;
    }
    for (int i = 0; i < d_best_size; i++) {
        printf("%d -> ", max_clique[i]);
    }
    std::cout << "\n\nTime difference = " << ms.count() << "[ms] " << ms.count() / 1000 << "[s] " << ms.count() / 60000 << "[m] " << std::endl;
Error:
    cudaFree(dev_max_clique);
    cudaFree(d_n);
    free(max_clique);
    return cudaStatus;
}


