#ifndef KMEANS_WRAP
#define KMEANS_WRAP



#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <limits>
#include <iomanip>
#include <chrono>
#include "clusters.h"
/*
    intended interactions 
    user
    |
    | - - - k_means
            |
            | - - - DataPoint
            | - - - Clust
                    |
                    | - - - DataPoint 
*/
/*
   the k_means class is a wrapper for the cluster 
   class. it is a middle man / API for the algorithm.

   the template T is to be passed down to k_means. which 
   k_means passes down to the DataPoint class. 

   see :: clust & dataPoint :: classes for more information 
*/
template <class T>
class k_means
{
    int total_runs; 
    double convergence; 
    int k_value; 
    int max_iterations; 
   
    int current_run; // potentially get rid of
    int best_run_indx; 
    double best_run_iter_sse; 
    
    std::vector<dataPoint<T>> data_set;  
    std::vector<T> maxDataVector; 
    std::vector<T> minDataVector; 
    clust<T>* cluster_list; 
    clust<T>* best_run_clust; 
    // ============================================================= helper methods
   
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
   
   
   
    // - - - - - - generates dataPoint structs & populates data_set
    void populateDataSet(const std::string& line, int& id, bool& firstRun)
    {           
         // read rows element by element
        std::istringstream lineStream(line);
        std::string elements; 
        std::vector<double> featureVector;

        int elmNum =0; 
        while(getline(lineStream, elements, ' '))
        {
            // ---- skip empty elements 
            if (!elements.empty())
            {
                double elmToDec = std::stod(elements); 
                // ----- first run populates min/max vectors, other runs find min/max
                if(firstRun == true)
                {
                    maxDataVector.push_back(elmToDec);
                    minDataVector.push_back(elmToDec);
                }
                else
                {
                    // swap attributes if elm in vector is smaller
                    if(maxDataVector.at(elmNum) < elmToDec)
                    {
                        maxDataVector.at(elmNum) = elmToDec; 
                    }
                    // swap attributes if elm in vector is greater
                    if(minDataVector.at(elmNum) > elmToDec)
                    {
                        minDataVector.at(elmNum) = elmToDec; 
                    }
                }
                featureVector.push_back(elmToDec);
                elmNum++; 
            }
        }

        // ---- skip empty vectors  
        if (!featureVector.empty())
        {
            std::string point_ID = "dataPoint " + std::to_string(id); 
        
            data_set.push_back(dataPoint<T>(std::move(featureVector), 0.0, point_ID)); 
            id++; 
        }              
        
        if(firstRun == true) {firstRun = false; }

    }
  
  
  
    // - - - - - -  checks if the current cluster's rand centroid is not already used
    bool checkUnique(int currentCluster, int centroidIndx )
    {
        const std::string& dataPointID = data_set.at(centroidIndx).getDataID(); 
        bool unique = true; 

        for(int i = 0; i < currentCluster; i++)
        {
            dataPoint<T> prevCentroids = cluster_list[i].getCentroid(); // !!!!!!!!! find a way way to avoid copying !!!!!!!!
                 
            if(dataPointID == prevCentroids.getDataID())
            {
                unique = false; 
                break; 
            }
        }

        return unique; 
    }
   
   
   
    // - - - - - finds the closest centroid for a given dataPoint
    int closestCentroid(int currentPoint)
    {
        // using largest value so that it can be replaced with increasingly smaller values
        double bestSqrDifference = std::numeric_limits<double>::max(); 
        int bestFitIndex =0; 
        std::vector<T> currDataVector = data_set.at(currentPoint).getDataVector(); // !!!!!!!!!!!!! find a way to avoid copying !!!!!!!!


        for (int i = 0; i < k_value; i++)
        {
            clust<T>& currClust = cluster_list[i]; 
            const std::vector<T>& dataVector = currClust.getCentroid().getDataVector(); 
            double totalSqrDifference =0.0; 

            // ---- finds the total difference between dataPoint features and centroid features
            //      squaring it so that I have a positive number at the end
            for(int featIndx =0; featIndx < dataVector.size(); featIndx++)
            {
                double diff = dataVector.at(featIndx) - currDataVector.at(featIndx);
                totalSqrDifference += (diff * diff);                
            }
            // ----- replace bestDistance & bestFit with closer centroid 
            if(totalSqrDifference < bestSqrDifference)
            {
                bestSqrDifference = totalSqrDifference; 
                bestFitIndex = i; 
            }
        }
        // returns the cluster_list index of the cluster with the best fit
        return bestFitIndex; 
    }
  
  
  
    // - - - - - checks for and calculates convergence 
    bool tryConvergence(std::vector<double>& iter_sse_vector, int current_iteration, double iterSSE)
    {
        bool converged = false; 
        
        // ---- convergence check for :: i > 0 or i == 0
        if(current_iteration > 0)
        { 
            // (SSE^t-1 - SSE^t) / SSE^t-1
            double prevSSE = iter_sse_vector.at(current_iteration - 1);
            double currentSSE = iter_sse_vector.at(current_iteration); 
            double convCheck = (prevSSE - currentSSE) / prevSSE;

            if (convCheck < convergence)
            {
                converged = true; 
            }
        }
        else 
        {
            if(iterSSE < convergence) 
            {              
                converged =  true; 
            }
        }

        // ---- if converged, compare current iteration to algorithm's best iteration
        if(best_run_iter_sse > iterSSE && converged == true)
        {
            // sets values for the current best 'run' 
            best_run_iter_sse = iterSSE; 
            best_run_indx = current_run; 
            // safe to transfer 'best cluster' ownership ; cluster_list will be reset anyways 
            best_run_clust = std::move(cluster_list);
        } 

        return converged; 
    }
   
   
   
    // - - - - - prints each iteration SSE value
    void printIterationSSE(const std::vector<double>& sse_values)
    {
        for(int i =0; i < sse_values.size(); i++)
        {
            std::cout << "iteration " << (i+1) << " SSE :: " << sse_values.at(i) <<std::endl; 
        }
    }
   
   
   
    // - - - - - prints iteration SSE to an output file 
    void logIterationSSE(const std::vector<double>& sse_values, std::ofstream& path_to_output)
    {
        for(int i =0; i < sse_values.size(); i++)
        {
            path_to_output << "iteration " << (i+1) << " SSE :: " << sse_values.at(i) <<std::endl; 
        }
    }
   
   
   
    // - - - - -  finds and sets cluster's mean centroids
    void genMeanCentroid()
    {
        // ------ reset clusters with new centroid
        for(int i =0; i < k_value; i++)
        {   
            clust<T>& currClust = cluster_list[i]; 
         
            // generate the dataPoint for the new centroid
            std::vector<T> meanCentroid = currClust.centroidUsingMean(data_set); 
            std::string id = "Mean Centroid " + std::to_string(i); 
            
            dataPoint<T> newPoint(meanCentroid, 0.0, id); // !!!!!!!!!!!!!!!! remove this later on !!!!!!!!!!! 

            // reset cluster and assign next centroid 
            currClust.resetCluster(); 
            currClust.assignCentroid(newPoint); // !!!!!!!!!!!!! find a way to build dataPoint within method call !!!!!!!!!!
        }
    } 



   // - - - -  generates the iteration SSE  
    double genIterationSSE()
    {
        double totalIterationSSE = 0.0;
        for(int i =0; i< k_value; i++)
        {
            clust<T>& currClust = cluster_list[i]; 
            currClust.genSSE(data_set); 
            totalIterationSSE += currClust.getClassLevelSSE(); 
        }
        return totalIterationSSE; 
    }
    
    // ============================================================== configuration methods
    // - - - - - min/max data normalization
    void normalizeData()
    {

        // ------ goes through each feature vector
        for(int dataVector =0; dataVector < data_set.size(); dataVector++)
        {
            dataPoint<T>& currPoint = data_set.at(dataVector);
            std::vector<T> currFeatVect = currPoint.getDataVector(); 
            
            // ----- goes through each elm in feature vector
            for(int vectorElm = 0; vectorElm < currFeatVect.size(); vectorElm++ )
            {
                double elm = currFeatVect.at(vectorElm); 
                double maxVal = maxDataVector.at(vectorElm); 
                double minVal = minDataVector.at(vectorElm); 
                
                // x' = x - min(x) / max(x) - min(x)
                double normalizedElm = 0.5; 
                if((maxVal - minVal) != 0)
                {
                    normalizedElm = (elm - minVal) / (maxVal - minVal);
                }

                currFeatVect.at(vectorElm) = normalizedElm; 
            }
            currPoint.replaceDataVector(currFeatVect); 
        }
    }
   
    // - - - - - - takes path to file & fills data_set vector | so that I don't have to keep reading from file 
    void fillDataSet(const std::string&  data_file)
    {
        std::string line; 
        std::ifstream fn(data_file); 
        bool isFirstLine = true; 
        int id = 0; 
        bool firstRun = true; 

        // ---- check if file is open
        if(!fn.is_open())
        {
            std::cout << "ERROR :: file is not opening"<<std::endl; 
            std::cout << "INFO  :: file path = " << data_file <<std::endl; 
            exit(EXIT_FAILURE);
        }

        // -------- read lines from file
        while(getline(fn, line))
        {
            // ---- first line = header, following lines = data 
            if(isFirstLine == false)
            {
               populateDataSet(line, id, firstRun); 
            }
            else 
            {
                id++; 
                isFirstLine = false;    
            }
        }
       
        fn.close(); 
    }



   // - - - - - - finds the cluster's random data clusters
    void randCentroid()
    {
        int currentCluster = 0; 
        // -------- runs until all K clusters have centroids
        while (currentCluster < k_value)
        {
            int centroidIndx = rand() % data_set.size(); 

            // ---- first cluster, no need to compare
            if(currentCluster == 0)
            {
                clust<T>& currClust = cluster_list[currentCluster];                 
                currClust.assignCentroid(data_set.at(centroidIndx)); 
                currentCluster++; 
            }
            else // ---- compare with other clusters, no repeated centroids 
            {               
                // ---- assign centroid if unique
                if(checkUnique(currentCluster, centroidIndx))
                {
                    clust<T>& currClust = cluster_list[currentCluster]; 
                    currClust.assignCentroid(data_set.at(centroidIndx));  
                    currentCluster++;
                }
            }
        }

    }



   // - - - - - - - assign data from data_set to cluster 
    void assignClustData()
    {
        int currentPoint = 0;

        // ----- loops through data set to find closest centroid per dataPoint
        //       Clust saves the dataPoint index from data_set, not the dataPoint itself
        while(currentPoint < data_set.size())
        {
            int bestFitIndex = closestCentroid(currentPoint);    
            clust<T>& bestClust = cluster_list[bestFitIndex];  
            bestClust.assignData(currentPoint); 
            currentPoint++; 
        } 
        
    }    



   // - - - - - initializes the whole algorithm 
    void initKmeans(const std::string& file_path)
    {
        for(int i =0; i < k_value; i++)
        {
            clust<T> clust; 
            cluster_list[i] = std::move(clust);
        }

        fillDataSet(file_path);
        randCentroid();
        assignClustData();  
        current_run++; 
    }



   // - - - - - - resets to base between runs 
    void resetKmeans()
    {
        for(int i =0; i < k_value; i++)
        {
            clust<T> clust; 
            cluster_list[i]= std::move(clust);
        }
        randCentroid(); 
        assignClustData(); 
    }
    
    
    // ============================================ some logic
    // - - - - - run without logging 
    void startRun()
    {  
        int current_iteration = 0; 
        std::vector<double> all_iteration_sse; 
        bool hasConverged = false; 

        while (current_iteration < max_iterations && !hasConverged)
        {
            genMeanCentroid(); 
            assignClustData(); 
            double iterationSSE = genIterationSSE();
            all_iteration_sse.push_back(iterationSSE);
            
            // ---- convergence check 
            hasConverged = tryConvergence(all_iteration_sse, current_iteration, iterationSSE);
            current_iteration++; 
            
        }
        printIterationSSE(all_iteration_sse); 
        
    }



    // - - - - - - run and log
    void startLogRun( std::ofstream& path_to_output)
    {  
        int current_iteration = 0; 
        std::vector<double> all_iteration_sse; 
        bool hasConverged = false; 

        while (current_iteration < max_iterations && !hasConverged)
        {
            genMeanCentroid(); 
            assignClustData(); 
            double iterationSSE = genIterationSSE();
            all_iteration_sse.push_back(iterationSSE);
            
            // ---- convergence check 
            hasConverged = tryConvergence(all_iteration_sse, current_iteration, iterationSSE);
            current_iteration++; 
            
        }
        printIterationSSE(all_iteration_sse); 
        logIterationSSE(all_iteration_sse, path_to_output);
        
    }



    public :
    
    
    // ================================================================ User API
    // - - - - - - runs the algorithm 
    void runAlg(const std::string& file_path)
    {
        // -------- configure kmeans; start run; reset cluster_list; reset kmeans for new run
        initKmeans(file_path); 

        auto start = std::chrono::high_resolution_clock::now();
        while (current_run <= total_runs)
        {         
            std::cout << "\nrun " << current_run << "\n------------------------\n"<<std::endl;            
            startRun(); 
            resetKmeans();
            current_run++; 
        } 
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> seconds_taken = end - start;

        // print best run & it's iteration SSE 
        std::cout << "\nRun "<< best_run_indx 
                  << " is best run :: iteration SSE\t" << best_run_iter_sse <<std::endl; 
        std::cout << "total time taken for algorithm :: " << seconds_taken.count() << " s" << std::endl;
    }
    

    
    // - - - - - - normalizes the data and runs the algorithm
    void normRunAlg(const std::string& file_path)
    {
        // -------- configure kmeans; start run; reset cluster_list; reset kmeans for new run
        initKmeans(file_path); 
        normalizeData(); 

        auto start = std::chrono::high_resolution_clock::now();
        while (current_run <= total_runs)
        {         
            std::cout << "\nrun " << current_run << "\n------------------------\n"<<std::endl;            
            startRun(); 
            resetKmeans();
            current_run++; 
        } 
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> seconds_taken = end - start;

        // print best run & it's iteration SSE 
        std::cout << "\nRun "<< best_run_indx 
                  << " is best run :: iteration SSE\t" << best_run_iter_sse <<std::endl; 
        std::cout << "total time taken for algorithm :: " << seconds_taken.count() << " s" << std::endl;

    }



    // - - - - - - run the algorithm and logs
    void runAndLogAlg(const std::string& path_to_data, const std::string& path_to_output)
    {
        // -------- configure kmeans; start run; reset cluster_list; reset kmeans for new run
        initKmeans(path_to_data); 
        std::ofstream log_to_Output(path_to_output); 
        
        if(!log_to_Output.is_open())
        {
            std::cout << "ERROR :: file is not opening"<<std::endl; 
            std::cout << "INFO  :: file path = " << path_to_data <<std::endl; 
            exit(EXIT_FAILURE);
        }

        auto start = std::chrono::high_resolution_clock::now();
        while (current_run <= total_runs)
        {         
            // ---- console output then file output
            std::cout << "\nrun " << current_run << "\n------------------------\n"<<std::endl;   
            log_to_Output << "\nrun " << current_run << "\n------------------------\n"<<std::endl;     

            startLogRun(log_to_Output);            
            resetKmeans();
            current_run++; 
        } 
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> seconds_taken = end - start;


        // ---- formatting to keep it consistent with console output
        std::ostringstream oss;

        oss << "\nRun " << best_run_indx
        << " is best run :: iteration SSE\t"
        << std::fixed << std::setprecision(4) << best_run_iter_sse
        << "\n";
        std::string outputMessage = oss.str();
        
        // ---- outputs to console then to file
        std::cout << outputMessage; 
        log_to_Output << outputMessage; 
        log_to_Output << "total time taken for algorithm :: " << seconds_taken.count() << " s" <<std::endl; 

        log_to_Output.close(); 
        
    }


    
    // - - - - - - normalizes the data, runs the algorithm, and logs
    void normRunAndLogAlg(const std::string& file_path)
    {

    }

    
    // ==================================================== constructor 
    k_means(int k, int runs, int max_iterations, double convergence)
    {
        srand(time(0)); 
        validateData(k, max_iterations, convergence, runs);
   
        current_run = 0; 
        best_run_indx = 0; 
        best_run_iter_sse = std::numeric_limits<double>::max(); 

        data_set = std::vector<dataPoint<T>>{};   
        cluster_list = new clust<T>[k_value];
        best_run_clust = new clust<T>[k_value]; 
    
    }
    ~k_means()
    {
        delete[] cluster_list; 
        delete[] best_run_clust; 
    }
};
     /*
            :: NOTES TO SELF ::

            Use references wherever possible:
            dataType & 
            const dataType & 
        
            pass by reference with every structure you can
            pass by value means copying, for large K values or large data this is a problem
            pass by reference means we are using the original value, so there is no copying
            if K is large, and the algorithm is copying at least once per K, lots of time and space is used
            references will speed algorithm up considerably 


            Std::move() is cool, here is how it works

            variable X essentially 'preps' data or messes with whatever it is holding
            variable Y wants whatever the data X hold, but only after it is prepped
            once X preps the data, we don't need it
            std::move() essentially reassigns ownership of the data
            this avoids having to copy data from X to Y (Y = X) which takes a lot of time with large data
            X = std::move(Y) essentially means 
            X prepared the data, and X no longer needs the data, so it is handing ownership over to Y
            
            do not use X after std::move() without reinitializing it
            std::move() tells us that X's job is done and it is safe to transfer ownership and trash
    
        */ 
#endif