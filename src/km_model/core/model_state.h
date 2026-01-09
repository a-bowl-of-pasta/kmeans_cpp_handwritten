/*
    model_state holds the current state of the algorithm
    it has all the key data and passes it around 
*/
#ifndef MODEL_STATE
#define MODEL_STATE

#include "../clusters.h"

template <class T>
struct model_state
{
    int total_runs; 
    double convergence; 
    int k_value; 
    int max_iterations; 
   
    int current_run; 
    int best_run_indx; 
    double best_run_iter_sse; 
    double best_chi_score; 
    double best_shiloette; 
    double best_jaccard_score; 
    double best_rand_indx_score;
    
    std::vector<dataPoint<T>> data_set;  
    std::vector<T> maxDataVector; 
    std::vector<T> minDataVector; 
    std::vector<clust<T>> cluster_list; 
    std::vector<clust<T>> best_run_clust; 
    std::vector<int> target_clusters; 

    // ========================  basic update methods
    
    // - - - - incriment current run
    void currentRun_increase()
        {current_run ++;}

    // - - - - updates the best run
    void updateBestRun()
    {
        best_run_clust = cluster_list; 
    }

    // - - - - checks if 'best_chi' should be replaced
    void checkBetterChi(double potential_Chi)
    {   
        if(potential_Chi > best_chi_score)
            best_chi_score = potential_Chi; 

    }
    
    // - - - - checks if 'best_shiloette' should be replaced
    void checkBetterShil(double potential_shiloette)
    {
        if(potential_shiloette > best_shiloette)
            best_shiloette = potential_shiloette; 
    }

    // - - - - chekcs if 'best_jaccard_indx' should be replaced
    void checkBetterJaccard(double potential_jacc)
    {
        if(potential_jacc > best_jaccard_score)
            { best_jaccard_score = potential_jacc; }
    }

    // - - - - check if 'best_rand_indx_score' should be repalced
    void checkBetterRandIndx(double potential_rand_indx)
    {
        if(potential_rand_indx > best_rand_indx_score)
            { best_rand_indx_score = potential_rand_indx; }

    }

    // ======================== other useful methods

    // - - - - - - validates and assigns the main values for the constructor 
    void validateData(int k, int iter, double converg, int runs)
    {
        // ----- validates and assigns variables 
        if (k <= 1 || iter <= 0 || runs <= 0) 
        {
           std::cout << "ERROR :: Invalid arguments | values entered : k = " << k 
                     << ", iterations = " << iter
                     << ", total runs ="  << runs <<std::endl; 
           std::cout << "INFO  :: must be : k > 1, iterations > 0, runs > 0 " <<std::endl; 
            exit(EXIT_FAILURE);
        }

        k_value = k; 
        max_iterations = iter; 
        convergence = converg;
        total_runs = runs; 
    }
   
    model_state(int k_val, int total_runs, int max_iter, double converg)
    {
        validateData(k_val, max_iter, converg, total_runs);
        
        current_run = 0; 
        best_run_indx = 0; 
        best_run_iter_sse = std::numeric_limits<double>::max(); 
        best_chi_score = std::numeric_limits<double>::min(); 
        best_shiloette = std::numeric_limits<double>::min(); 

        data_set = std::vector<dataPoint<T>>{};   
        target_clusters = std::vector<int>{}; 
        cluster_list = std::vector<clust<T>>{};
        best_run_clust = std::vector<clust<T>>{};


        cluster_list.reserve(k_val);
        best_run_clust.reserve(k_val);

    }

};


#endif