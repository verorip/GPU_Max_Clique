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



constexpr int N = 30;
constexpr int RATE = 30;
// flag =0 solo sequenziale, flag =1 entrambi, flag >1 solo parallelo
constexpr int flag = 1;

//per sincronizzare eseguo atomicexec()
__device__ int d_best_size[N];
__device__ int dev_max_clique[N * N];
__device__ int deepest[N];


cudaError_t clique_launcher(const std::vector<int>& degrees, const std::vector<int>& neighbours, const std::vector<int>& indexes);




void rec_seq_clique(const std::vector<int> &degrees, const std::vector<int>&neighbours, std::deque<int> &intersection,const std::vector<int>&indexes, std::array<int, N> &max_clique_seq, std::array<int,N> &tmp, int &best_size, int tmp_index, int current) {
    int crt;
    std::deque<int> local_neight(degrees[current]);
    
   
    
    for (int i = 0; i < degrees[current]; i++) {
        local_neight[i] = neighbours[indexes[current] + i];
    }
    std::sort(local_neight.begin(), local_neight.end());
    std::deque<int> inters_local;
    if (tmp_index > 1) {
        std::set_intersection(intersection.begin(), intersection.end(),
            local_neight.begin(), local_neight.end(),
            back_inserter(inters_local));
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
        
        rec_seq_clique(degrees, neighbours, std::deque<int>(0), indexes, max_clique_seq, tmp, best_size, 1, i);
        
    }
}

std::vector<int> create_graph(std::vector<int>& degrees, int MaxNeightbours) {
    std::vector<int> temp_neighbours(N * MaxNeightbours);

    int sizeOfN = 0;
    int r = 0;
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {

            r = rand() % 100;
            if (r < RATE) {
                if (degrees[i] >= MaxNeightbours || degrees[j] >= MaxNeightbours) printf("Troppi vicini, balzo\n");
                else {
                    printf("%d, %d, %d, %d\n", i, j, (i * MaxNeightbours) + degrees[i], (j * MaxNeightbours) + degrees[j]);
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

    int f = 0;
    int count = 0;
    for (int i = 0; i < indexes.size(); i++) {
        for (int j = 0; j < degrees[i]; j++) {
            std::cout << neighbours[indexes[i]+j] << ' ';
        }
        std::cout << "  //  ";
    }
    std::cout << '\n';

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    if (flag <= 1) {
        printf("\n Inizio il sequenziale: \n");
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
        
        cudaError_t cudaStatus = clique_launcher(degrees, neighbours, indexes);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "kernel fail!");
            return 1;
        }
    }    
}

/*__device__*/ int parallel_intersection(int *arr1, int *arr2, int x, int y, int *p) {
    int* tmp;
    if (y > x)
        tmp = new int[x];
    else
        tmp = new int[y];
    //int* tmp = new int[y ^ ((x ^ y) & -(x < y))];
    int i = 0, j = 0, count = 0;
    while (i < x && j < y) {
        if (arr1[i] < arr2[j])
            i++;
        else if (arr2[j] < arr1[i])
            j++;
        else /* if arr1[i] == arr2[j] */
        {
            tmp[count] = arr1[i];
            i++;
            j++;
            count++;
        }
    }
    p = new int[count];
    for (int k = 0; k < count; k++) {
        p[k] = tmp[k];
    }
    delete[] tmp;
    return count;

}

/*__device__*/ void rec_paral(int* dev_degrees, int* dev_neighbours, int* dev_indexes, int* intersection, int inter_size, int *tmp, int tmp_indexes, int current, int global_idx) {
    int* next_inter = new int[0];
    int size=0, i = 0;
    deepest[global_idx] = tmp_indexes;
    while (i < inter_size) {
        size = parallel_intersection(intersection, &dev_neighbours[dev_indexes[intersection[i]]], inter_size, dev_degrees[current], next_inter);
        tmp[tmp_indexes] = intersection[i];
        if(tmp_indexes<2)
            rec_paral(dev_degrees, dev_neighbours, dev_indexes, next_inter, size, tmp, tmp_indexes+1, intersection[i], global_idx);
        i++;
    }
    
    if (tmp_indexes > d_best_size[global_idx]) {
        d_best_size[global_idx] = tmp_indexes;
        for (int j = 0; j < tmp_indexes; j++) {
            dev_max_clique[(N * (global_idx)) + j] = tmp[j];
        }      
    }
    
}


/*__global__*/ void paral_max_clique(int *dev_degrees,int *dev_neighbours, int* dev_indexes, int n_size)
{
    /*int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int local_idx = threadIdx.x;*/
    int global_idx = 0;
    int tmp[N];
    for (int i = 0; i < N; i++) {
        tmp[i] = -1;
    }
    int tmp_index = 1;
    tmp[0] = global_idx;
    int *intersection= new int[dev_degrees[global_idx]];
    for (int i = 0; i < dev_degrees[global_idx]; i++) {
        intersection[i] = dev_neighbours[dev_indexes[global_idx]+i];
    }    
    rec_paral(dev_degrees, dev_neighbours, dev_indexes, intersection, dev_degrees[global_idx], tmp, 1, global_idx, global_idx);
    
    delete[] intersection;

}



//qui eseguirò il codice parallelo
cudaError_t clique_launcher(const std::vector<int>& degrees, const std::vector<int>& neighbours, const std::vector<int>& indexes)
{
    int to_ret[N*N] = { -1 };
    int *dev_degrees;
    int *dev_indexes;
    int* dev_neighbours;
    int tmp[N];
    int sz = neighbours.size();
    int d[N];
    int in[N];

    for (int i = 0; i < N * N; i++) {
        to_ret[i] = -1;
    }
    for (int i = 0; i < N; i++) {
        tmp[i] = -1;
    }
    int* n = (int*)malloc(neighbours.size() * sizeof(int));
    cudaError_t cudaStatus;

    for (int i = 0; i < N; i++) {
        d[i] = degrees[i];
        in[i] = indexes[i];
    }
    for (int i = 0; i < neighbours.size(); i++) {
        n[i] = neighbours[i];
    }
    paral_max_clique(d, n,in, 0);
    /*
    cudaStatus = cudaMalloc((void**)&dev_degrees, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_best_size, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&deepest, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_max_clique, N*N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    
    cudaStatus = cudaMalloc((void**)&dev_neighbours, neighbours.size() * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_indexes, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_degrees, d, N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_neighbours, n, neighbours.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_indexes, in, N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpyToSymbol(dev_max_clique, to_ret, N * N* sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpytosymbol failed!", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaMemcpyToSymbol(d_best_size, tmp, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpytosymbol failed!", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaMemcpyToSymbol(deepest, tmp, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpytosymbol failed!", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    printf("\n\navvio kernel\n\n ");
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //qua lancio il kernel
    paral_max_clique<<<1, N>>> (dev_degrees, dev_neighbours, dev_indexes, sz);
    
    //controllo errori nel lancio dle kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "max_clique_parallel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<float, std::milli> ms = end - begin;
    std::cout << "\n\nTime difference = " << ms.count() << "[ms] " << ms.count() / 1000 << "[s] " << ms.count() / 60000 << "[m] " << std::endl;

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpyFromSymbol(to_ret, dev_max_clique, N * N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyFromSymbol failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpyFromSymbol(tmp, d_best_size, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyFromSymbol failed!");
        goto Error;
    }
    printf("\n d_best: \n");
    for (int i = 0; i < N; i++) {
        printf("%d -> ", tmp[i]);
    }
    int i = 0, out = 0, max=0;
    while (i < N && out == 0) {
        max = tmp[i] > max ? i : max;
        i++;
    }
    
    printf("\n clique:\n");
    for (i = 0; i < tmp[max]; i++) {

        printf("%d -> ", to_ret[(N*max)+i]);
    }
    /*for (i = 0; i < N * N; i++) {
        printf("%d -> ", to_ret[i]);
    }

Error:
    cudaStatus = cudaMemcpyFromSymbol(tmp, deepest, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyFromSymbol failed! %s", cudaGetErrorString(cudaStatus));
    }
    printf("deepest:\n");
    for (i = 0; i < N; i++) {

        printf("%d: %d -> ", i,to_ret[i]);
    }
    cudaFree(d_best_size);
    cudaFree(dev_max_clique);
    cudaFree(deepest);
    cudaFree(dev_degrees);
    cudaFree(dev_neighbours);
    cudaFree(dev_indexes);*/
        
    return cudaStatus;
}


