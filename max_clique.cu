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



constexpr int N = 128;
constexpr int RATE = 45;
// flag =0 solo sequenziale, flag =1 entrambi, flag >1 solo parallelo
constexpr int flag = 1;

thrust::device_vector<int> dev_max_clique(N, -1);



cudaError_t clique_launcher(const std::vector<int>& degrees, const std::vector<int>& neighbours, const std::vector<int>& indexes);


__host__ void rec_seq_clique(const std::vector<int> &degrees, const std::vector<int>&neighbours, std::deque<int> &intersection,const std::vector<int>&indexes, std::array<int, N> &max_clique_seq, std::array<int,N> &tmp, int &best_size, int tmp_index, int current) {
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
__host__ void seq_clique(const std::vector<int> &degrees, const std::vector<int>&neighbours, const std::vector<int> &indexes, std::array<int, N> &max_clique_seq, std::array<int,N> &tmp, int &best_size) {
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


void rec_par_clique(const std::vector<int>& degrees, const std::vector<int>& neighbours, std::deque<int>& intersection, const std::vector<int>& indexes, std::array<int, N>& tmp, int& best_size, int tmp_index, int current){

}

void parallel_clique(const std::vector<int>& degrees, const std::vector<int>& neighbours, const std::vector<int>& indexes, std::array<int, N>& tmp, int& best_size) {
    best_size = 0;
    for (int i = 0; i < N; i++) {
        tmp[0] = i;
        std::deque<int> d = std::deque<int>(0);
        rec_par_clique(degrees, neighbours, d, indexes, tmp, best_size, 1, i);

    }
}

//qui eseguirò il codice parallelo
__host__ cudaError_t clique_launcher(const std::vector<int>& degrees, const std::vector<int>& neighbours, const std::vector<int>& indexes)
{
    cudaError_t cudaStatus = cudaSuccess;

    
    int d_best_size = 1;
    parallel_clique(degrees, neighbours, indexes, std::array<int,N>(), d_best_size);
    return cudaStatus;
}


