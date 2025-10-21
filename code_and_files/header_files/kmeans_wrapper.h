#ifndef KM_WRAPPER
#define KM_WRAPPER

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include "kmeans.h"

template <class T>
class k_means_controller 
{
    int total_runs; 
    double convergence; 
    int k_value; 
    int max_iterations; 
   
    int current_run; 

    std::vector<double> iteration_SSE; 
    std::vector<double> run_SSE; 
    std::vector<dataPoint<T>> dataSet;  
    std::vector<k_means<T>> clusterList; 

    // = = = = = = = = = = = = private helper methods
    std::vector<double> genDataPoint(std::string line)
    {           
         // read rows element by element
        std::istringstream lineStream(line);
        std::string elements; 
        std::vector<double> pointVector;

        while(getline(lineStream, elements, ' '))
        {
            // skip empty elements 
            if (!elements.empty())
            {
                pointVector.push_back(std::stod(elements));
            }
        }
        return pointVector; 
    }

    public :
    // = = = = = = = = = = = = = = = = = = = = = = = getters and setters
    void centroidUpdate()
    {

    }
 
    void getClusters()
    {

    }
    void getIterationSSE()
    {

    }
    void getBestRunSSE()
    {

    }
    // = = = = = = = = = = = = = = = = = = = = = = = = configuration methods
    void fillDataSet(std::string data_file)
    {
        std::string line; 
        std::ifstream fn(data_file); 
        int id = 0; 

        // -------------- check if file is open
        if(!fn.is_open())
        {
            std::cout << "ERROR :: file is not opening"<<std::endl; 
            std::cout << "INFO  :: file path = " << data_file <<std::endl; 
            exit(EXIT_FAILURE);
        }

        // -------------------- read lines from file
        while(getline(fn, line))
        {

            std::vector<double> pointVector = genDataPoint(line); 
            // --------------------- skip empty vectors
            if (!pointVector.empty())
            {
                std::string point_ID = "dataPoint " + id; 
                dataSet.push_back(dataPoint<T>(pointVector, 0.0, point_ID)); 
                id++; 
            }
        }

        fn.close(); 
    }
    void centroidInit()
    {
        int currentCluster = 0; 
        while (currentCluster < clusterList.size())
        {
            int centroidPos = rand() % dataSet.size(); 

            if(currentCluster == 0)
            {
                k_means<T>& currClust = clusterList.at(currentCluster); 
                dataPoint<T> chosenCentroid = dataSet.at(centroidPos); 
                
                currClust.setCentroid(chosenCentroid); 
                currentCluster++; 
            }
            else
            {
                std::string dataPointID = dataSet.at(centroidPos).getDataID(); 
                bool unique = true; 

                for(int i = 0; i < currentCluster; i++)
                {
                    dataPoint<T> prevCentroids = clusterList.at(i).getCentroid();
                 
                    if(dataPointID == prevCentroids.getDataID())
                    {
                        unique = false; 
                        break; 
                    }
                }
                if(unique)
                {
                    k_means<T>& currClust = clusterList.at(currentCluster); 
                    dataPoint<T> chosenCentroid = dataSet.at(centroidPos); 
                    
                    currClust.setCentroid(chosenCentroid); 
                    currentCluster++;
                }
            }
        }

    }
    
    // = = = = = = = = = = = = = = = = = = = = = = = main user methods
    double genTotalSSE()
    {

    }
 
    void startRun()
    {
       
    }

    // = = = = = = = = = = = = = constructor
    k_means_controller(int runs, double convergence, int k, int max_iterations)
    {
        srand(time(0)); 
        if(k <= 1)
        {
            std::cout << "ERROR :: K must be greater than 1" << std::endl; 
            std::cout << "INFO :: k = " << k << std::endl;
            exit(EXIT_FAILURE); 
        }
        total_runs = runs; 
        this->convergence = convergence; 
        k_value = k; 
        this->max_iterations = max_iterations;
        current_run = 0; 

        iteration_SSE = std::vector<double>{}; 
        run_SSE = std::vector<double>{}; 
        dataSet = std::vector<dataPoint<T>>{};  
        clusterList = std::vector<k_means<T>>{};         

        for(int i =0; i < k; i++)
        {
            k_means<T> clust; 
            clusterList.push_back(clust);
        }
    
    }

};



#endif