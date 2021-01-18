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


constexpr int N = 60;
constexpr int RATE = 60;
// flag =0 solo sequenziale, flag =1 entrambi, flag >1 solo parallelo
constexpr int flag = 1;





cudaError_t clique_launcher(const std::vector<int>& degrees, const std::vector<int>& neighbours, const std::vector<int>& indexes);

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


void rec_par_clique(const std::vector<int>& degrees, const std::vector<int>& neighbours, thrust::host_vector<int>& intersection, const std::vector<int>& indexes, std::array<int, N>& tmp, int &best_size, int tmp_index, int current, thrust::device_vector<int>& to_ret){
    //std::deque<int> inters_local;
    thrust::host_vector<int>::iterator r;
    thrust::host_vector<int> intersection_local(degrees[current]<=intersection.size()-1? degrees[current] : intersection.size()-1, -1);
    if (tmp_index > 1) {
        r=thrust::set_intersection(intersection.begin()+1, intersection.end(), neighbours.begin() + indexes[current], neighbours.begin() + indexes[current] + degrees[current], intersection_local.begin());
    }
    else {
        //inters_local = std::deque<int>(neighbours.begin() + indexes[current], neighbours.begin() + indexes[current] + degrees[current]);
        intersection_local = thrust::host_vector<int>(neighbours.begin() + indexes[current], neighbours.begin() + indexes[current] + degrees[current]);
        r = intersection_local.end();
    }
    int i = 0;
    //int d_t = thrust::distance(intersection_local.begin(),r);
    int d = r - intersection_local.begin();
    while (i<d && d - i + tmp_index > best_size) {

        //crt = intersection_local[i];
        tmp[tmp_index] = intersection_local[i];
        rec_par_clique(degrees, neighbours, intersection_local, indexes, tmp, best_size, tmp_index + 1, intersection_local[i], to_ret);
        i++;
    }

    if (tmp_index > best_size) {
        best_size = tmp_index;
        thrust::device_vector<int> t(tmp.begin(), tmp.begin()+tmp_index);
        thrust::copy(thrust::device, t.begin(), t.end(), to_ret.begin());
    }
}

void parallel_clique(const std::vector<int>& degrees, const std::vector<int>& neighbours, const std::vector<int>& indexes, std::array<int, N>& tmp, int& best_size, thrust::device_vector<int>&  to_ret) {
    best_size = 0;
    for (int i = 0; i < N; i++) {
        tmp[0] = i;
        thrust::host_vector<int>  d = thrust::host_vector<int>(0);
        rec_par_clique(degrees, neighbours, d, indexes, tmp, best_size, 1, i, to_ret);

    }
}

//qui eseguirò il codice parallelo
cudaError_t clique_launcher(const std::vector<int>& degrees, const std::vector<int>& neighbours, const std::vector<int>& indexes)
{
    cudaError_t cudaStatus = cudaSuccess;


    int d_best_size = 0;
    std::array<int, N> tmp;
    for (int i = 0; i < N; i++) {
        tmp[i] = -1;
    }
    thrust::device_vector<int> dev_max_clique(N, -1);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    parallel_clique(degrees, neighbours, indexes, tmp, d_best_size, dev_max_clique);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<float, std::milli> ms = end - begin;
    
    printf("\n clique seq lunga %d: \n", d_best_size);
    thrust::copy(dev_max_clique.begin(), dev_max_clique.begin()+d_best_size, std::ostream_iterator<int>(std::cout, "--> "));
    /*for (int i = 0; i < d_best_size; i++) {
        printf("%d -> ", dev_max_clique[i]);
    }*/
    std::cout << "\n\nTime difference = " << ms.count() << "[ms] " << ms.count() / 1000 << "[s] " << ms.count() / 60000 << "[m] " << std::endl;
    return cudaStatus;
}


