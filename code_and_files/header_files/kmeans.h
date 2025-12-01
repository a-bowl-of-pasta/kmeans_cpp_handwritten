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
#include <random>
#include "clusters.h"


/*
    dataBucket is a storage structure, it contains
    all the information needed throughout the algorithm
*/
template <class T>
struct dataBucket
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
    std::vector<clust<T>> cluster_list; 
    std::vector<clust<T>> best_run_clust; 

    // - - - - - - basic update methods
    void updateBestRun()
    {
        best_run_clust = cluster_list; 
        cluster_list.clear();
        cluster_list.reserve(k_value);

    }

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
   
    dataBucket(int k_val, int total_runs, int max_iter, double converg)
    {
        validateData(k_val, max_iter, converg, total_runs);
        
        current_run = 0; 
        best_run_indx = 0; 
        best_run_iter_sse = std::numeric_limits<double>::max(); 

        data_set = std::vector<dataPoint<T>>{};   
        cluster_list = std::vector<clust<T>>{};
        best_run_clust = std::vector<clust<T>>{};

        cluster_list.reserve(k_val);
        best_run_clust.reserve(k_val);

    }

};


/*
    h_io is the input/output class, it holds methods that do input/output
*/
template <class T>
class h_io
{

    public:
    // ============================================= methods that printer to console 

    // - - - - - simple message output
    void consoleOutput(std::string& output)
    {
        std::cout << output << std::endl; 
    }

    // - - - - - prints each iteration SSE value
    void printIterationSSE(const std::vector<double>& sse_values)
    {
        for(int i =0; i < sse_values.size(); i++)
        {
            std::cout << "iteration " << (i+1) << " SSE :: " << sse_values.at(i) <<std::endl; 
        }
    }
    
    // ============================================= methods that log to file

    // - - - - - prints iteration SSE to an output file 
    void logIterationSSE(const std::vector<double>& sse_values, std::ofstream& path_to_output)
    {
        for(int i =0; i < sse_values.size(); i++)
        {
            path_to_output << "iteration " << (i+1) << " SSE :: " << sse_values.at(i) <<std::endl; 
        }
    } 

    // - - - - - - simple log output
    void fileOutput(std::ofstream& file, std::string output)
    {
        file << output << std::endl; 
    }

    h_io(){}
};


/*
    the h_backend class holds the backend logic for the k_means class
    it is for methods that performs calculations and data processing
    
*/
template <class T>
class h_backend
{

    std::vector<dataPoint<T>>& data_set;  
    std::vector<T>& maxDataVector; 
    std::vector<T>& minDataVector; 
    std::vector<clust<T>>& cluster_list; 

    public: 
    // =========================================================== calculation methods
   
    // - - - - finds the closest centroid to a point | not used outside this class
    int calcClosestCentroid(int currentPoint, int k_value) 
    {
        // using largest value so that it can be replaced with increasingly smaller values
        double bestSqrDifference = std::numeric_limits<double>::max(); 
        int bestFitIndex =0; 
        std::vector<T> currDataVector = data_set.at(currentPoint).getDataVector(); // !!!!!!!!!!!!! find a way to avoid copying !!!!!!!!


        for (int i = 0; i < k_value; i++)
        {
            clust<T>& currClust = cluster_list.at(i); 
            std::vector<T> dataVector = currClust.getCentroid().getDataVector(); 
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
        // returns the  cluster_list index of the cluster with the best fit
        return bestFitIndex; 
    }
  
    double calcIterationSSE(int k_value)
    {
        double totalIterationSSE = 0.0;
        for(int i =0; i< k_value; i++)
        {
            clust<T>& currClust = cluster_list.at(i); 
            currClust.genSSE(data_set); 
            totalIterationSSE += currClust.getClassLevelSSE(); 
        }
        return totalIterationSSE; 
    }
   

    // =========================================================== validation methods
    
    bool checkUnique( int currentCluster, int centroidIndx)
    {
        const std::string& dataPointID = data_set.at(centroidIndx).getDataID(); 
        bool unique = true; 

        for(int i = 0; i < currentCluster; i++)
        {
            dataPoint<T> prevCentroids = cluster_list.at(i).getCentroid(); // !!!!!!!!! find a way way to avoid copying !!!!!!!!
                 
            if(dataPointID == prevCentroids.getDataID())
            {
                unique = false; 
                break; 
            }
        }

        return unique; 
    }
   

    bool checkConvergance(dataBucket<T>& shared_data, std::vector<double>& iter_sse_vector, int current_iteration, double iterSSE)
    {
        bool converged = false; 
        
        // ---- convergence check for :: i > 0 or i == 0
        if(current_iteration > 0)
        { 
            // (SSE^t-1 - SSE^t) / SSE^t-1
            double prevSSE = iter_sse_vector.at(current_iteration - 1);
            double currentSSE = iter_sse_vector.at(current_iteration); 
            double convCheck = (prevSSE - currentSSE) / prevSSE;

            if (convCheck < shared_data.convergence)
            {
                converged = true; 
            }
        }
        else 
        {
            if(iterSSE < shared_data.convergence) 
            {              
                converged =  true; 
            }
        }

        // ---- if converged, compare current iteration to algorithm's best iteration
        if(shared_data.best_run_iter_sse > iterSSE && converged == true)
        {
            // sets values for the current best 'run' 
            shared_data.best_run_iter_sse = iterSSE; 
            shared_data.best_run_indx = shared_data.current_run; 

            // deep copy best run clusters
            shared_data.updateBestRun();
            
        } 

        return converged; 
    }
        

    // =========================================================== data processing methods
   
    // - - - - creates a new node (dataPoint) for the dataSet vector | not used outside this class
    void createDataSetNode( const std::string& line, int& id, bool& firstRun)
    {           
         // read rows element by element
        std::istringstream lineStream(line);
        std::string elements; 
        std::vector<double> featureVector_temp;

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
                featureVector_temp.push_back(elmToDec);
                elmNum++; 
            }
        }

        // ---- skip empty vectors  
        if (!featureVector_temp.empty())
        {
            std::string point_ID_temp = "dataPoint " + std::to_string(id); 
        
            data_set.push_back(dataPoint<T>(featureVector_temp, 0.0, point_ID_temp)); 
            id++; 
        }              
        
        if(firstRun == true) {firstRun = false; }

    }
    

    // - - - -  data normalization
    void calcNormalizedData()
    {

        // ------ goes through each feature vector
        for(int dataVector =0; dataVector < data_set.size(); dataVector++)
        {
            dataPoint<T>& currPoint_temp = data_set.at(dataVector);
            std::vector<T> currFeatVect_temp = currPoint_temp.getDataVector(); 
            
            // ----- goes through each elm in feature vector
            for(int vectorElm = 0; vectorElm < currFeatVect_temp.size(); vectorElm++ )
            {
                double elm = currFeatVect_temp.at(vectorElm); 
                double maxVal = maxDataVector.at(vectorElm); 
                double minVal = minDataVector.at(vectorElm); 
                
                // x' = x - min(x) / max(x) - min(x)
                double normalizedElm = 0.5; 
                if((maxVal - minVal) != 0)
                {
                    normalizedElm = (elm - minVal) / (maxVal - minVal);
                }

                currFeatVect_temp.at(vectorElm) = normalizedElm; 
            }
            currPoint_temp.replaceDataVector(currFeatVect_temp); 
        }
    }
   

    // - - - - -  assign data from  data_set to cluster 
    void fillClust(int k_value) 
    {
        int currentPoint = 0;

        // ----- loops through data set to find closest centroid per dataPoint
        //       Clust saves the dataPoint index from  data_set, not the dataPoint itself
        while(currentPoint < data_set.size())
        {
            int bestFitIndex = calcClosestCentroid(currentPoint, k_value);    
            clust<T>& bestClust = cluster_list.at(bestFitIndex);  
            bestClust.assignData(currentPoint); 
            currentPoint++; 
        } 
        
    }    
      
    
    // - - - - - - reads input file, 
    void createDataSetVector( const std::string&  data_file)
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
               createDataSetNode( line, id, firstRun); 
            }
            else 
            {
                id++; 
                isFirstLine = false;    
            }
        }
       
        fn.close(); 
    }


    // =========================================================== init strategies
   
    // - - - - random data point init
    void forgingInit(int k_value)
    {
        int currentCluster = 0; 

        // ---- better random numbers
        std::random_device randEngine; 
        std::uniform_int_distribution<int> numRange(0, data_set.size() -1); 
        
        // -------- runs until all K clusters have centroids
        while (currentCluster < k_value)
        {
            int centroidIndx = numRange(randEngine); 

            // ---- first cluster, no need to compare
            if(currentCluster == 0)
            {
                clust<T>& currClust = cluster_list.at(currentCluster);                 
                currClust.assignCentroid(data_set.at(centroidIndx)); 
                currentCluster++; 
            }
            else // ---- compare with other clusters, no repeated centroids 
            {               
                // ---- assign centroid if unique
                if(checkUnique( currentCluster, centroidIndx))
                {
                    clust<T>& currClust = cluster_list.at(currentCluster); 
                    currClust.assignCentroid(data_set.at(centroidIndx));  
                    currentCluster++;
                }
            }
        }

        fillClust(k_value); 
    }

    // - - - - random partition init
    void randomPartition(int k_value)
    {
        // picks random cluster form cluster list
        std::random_device randEngine; 
        std::uniform_int_distribution range(0, k_value-1);
        
        // ---- assign data to rand clust
        for(int i = 0 ; i < data_set.size(); i++)
        {
            int randNum = range(randEngine);
            
            clust<T>& randomChosenClust = cluster_list.at(randNum);
            randomChosenClust.assignData(i);  
        }   

        // ---- find mean featVec for each clust then assign as centroid
        for(int k = 0; k < k_value; k++)
        {

            clust<T>& currClust = cluster_list.at(k);
            std::vector<T> meanData_temp = currClust.genMeanDataVector(data_set);  
            
            std::string id_temp = "Mean Centroid " + std::to_string(k); 
            dataPoint<T> newPoint_temp(meanData_temp, 0.0, id_temp);

            currClust.assignCentroid(newPoint_temp);
        }

        // reassign data based off of the initial centroid
        fillClust(k_value); 
    }

    h_backend(dataBucket<T>& shared_data)
    : data_set(shared_data.data_set), 
      maxDataVector(shared_data.maxDataVector), 
      minDataVector(shared_data.minDataVector), 
      cluster_list(shared_data.cluster_list)
      {}
};


/*
   the k_means class is the main API, it is what the user
   will interact with. this class connects all the other ones together

   the template T is to be passed down to k_means. which 
   k_means passes down to the DataPoint class. 

   see :: clust & dataPoint :: classes for more information 
*/
template <class T>
class k_means
{
  
    h_io<T> io_manager;
    h_backend<T> backend_manager; 
    dataBucket<T> shared_data; 

    // ================================================= core methods

    // - - - - - sets the algorithm | initial set and reset between runs 
    void setKmeans(bool isInit, bool normalize = false, const std::string& file_path = "")
    {
        for(int i =0; i < shared_data.k_value; i++)
        {
            clust<T> clust; 
            shared_data.cluster_list.push_back(std::move(clust));
        }

        if(isInit)
        {
            backend_manager.createDataSetVector(file_path);
        }
        // ------ initialization strategy goes here 
    
         backend_manager.forgingInit(shared_data.k_value); 
        
        // ------ if initial run : current run 0 -> 1 && decide to normalize data 
        if(isInit)
        {
            if(normalize == true)
                { backend_manager.calcNormalizedData(); }
           
            shared_data.current_run++;    
        }
    }
    

    // - - - - - - finds and sets cluster's mean centroids
    void setNextCentroid()
    {
        // ------ reset clusters with new centroid
        for(int i =0; i < shared_data.k_value; i++)
        {   
            
            clust<T>& currClust = shared_data.cluster_list.at(i); 
            
            // store the mean data vector | call clust::genMeanDataVector
            std::vector<T> meanCentroid_temp = currClust.genMeanDataVector(shared_data.data_set); 
            std::string id_temp = "Mean Centroid " + std::to_string(i); 

            dataPoint<T> newPoint_temp(meanCentroid_temp, 0.0, id_temp); 

            // reset cluster and assign next centroid 
            currClust.resetCluster(); 

            currClust.assignCentroid(newPoint_temp); 
        }
    }  


    // - - - - - - backend for print | logic for when to print and what to print 
    void printRunDecision(bool printAllRuns, int& currentQuarter)
    {
          // prints all runs for small ranges of run | total_run <= 10  
            std::string mainOutput = "\nRun " + std::to_string(shared_data.current_run);    
            
            if(printAllRuns == true)
            {
                mainOutput.append("\n--------------\n");
                io_manager.consoleOutput(mainOutput); 

                startRun(printAllRuns); 
            }
            else // prints 6 runs for larger ranges of run | total_run > 10 
            {
                bool printCurrentRun = false; 
                int Qinterval; 

                // make total_runs even & cut into quarters 
                if (shared_data.total_runs % 2 == 0)
                    {Qinterval = (shared_data.total_runs) / 4;}
                
                else 
                    {Qinterval = (shared_data.total_runs -3) /4; }
                
                // varrifies that my printing interval isn't too small
                if(Qinterval < 2) 
                    {Qinterval == 2; }

                // prints 1st run | prints last run | prints quarter runs
                if(shared_data.current_run == 1 )
                { 
                    mainOutput.append("\n--------------\n");
                    io_manager.consoleOutput(mainOutput); 
                    printCurrentRun = true; 
                }

                else if(shared_data.current_run == shared_data.total_runs)
                {
                    mainOutput.append(" | last Run\n-------------\n");
                    io_manager.consoleOutput(mainOutput); 
                    printCurrentRun = true; 
                }

                else if(shared_data.current_run % Qinterval == 0)
                {    
                    mainOutput.append(" | Q" + std::to_string(currentQuarter) + "\n-------------\n");
                    io_manager.consoleOutput(mainOutput); 
                    printCurrentRun = true; 
                    currentQuarter++; 
                }

                // the run
                startRun(printCurrentRun);
            }
    }


    // - - - - - decides what to print / log
    void logRunDecision(bool logAllRuns, int& currentQuarter, std::ofstream& outFile)
    {
        std::string mainOutput = "Run " + std::to_string(shared_data.current_run); 

        // prints all runs for small ranges of run | total_run <= 10     
        if(logAllRuns == true)
        {
            mainOutput.append("\n--------------\n");

            io_manager.consoleOutput(mainOutput); 
            io_manager.fileOutput(outFile, mainOutput); 


            startLogRun(logAllRuns, outFile); 
        }
        else // prints 6 runs for larger ranges of run | total_run > 10 
        {
            bool printCurrentRun = false; 
            int Qinterval; 

            // make total_runs even & cut into quarters 
            if (shared_data.total_runs % 2 == 0)
                {Qinterval = (shared_data.total_runs) / 4;}
    
            else 
                {Qinterval = (shared_data.total_runs -3) /4; }
        
            // varrifies that my printing interval isn't too small
            if(Qinterval < 2) 
                {Qinterval == 2; }


            // prints 1st run | prints last run | prints quarter runs
            if(shared_data.current_run == 1 )
            { 
                mainOutput.append(" | first Run\n-------------\n");

                io_manager.consoleOutput(mainOutput); 
                io_manager.fileOutput(outFile, mainOutput); 
                printCurrentRun = true; 
            }
            
            else if(shared_data.current_run == shared_data.total_runs)
            {
                mainOutput.append(" | last Run\n-------------\n");
                
                io_manager.consoleOutput(mainOutput); 
                io_manager.fileOutput(outFile, mainOutput); 
                printCurrentRun = true; 
            }

            else if(shared_data.current_run % Qinterval == 0)
            {    
                mainOutput.append(" | Q" + std::to_string(currentQuarter) + "\n-------------\n");

                io_manager.consoleOutput(mainOutput); 
                io_manager.fileOutput(outFile, mainOutput); 
                printCurrentRun = true; 
                
                currentQuarter++; 
            }

            // starts run
            startLogRun(printCurrentRun, outFile);
        }
    }


    // - - - - - run without logging 
    void startRun(bool printRun)
    {  
        int current_iteration = 0; 
        std::vector<double> all_iteration_sse; 
        bool hasConverged = false; 

        while (current_iteration < shared_data.max_iterations && !hasConverged)
        {
            setNextCentroid(); 
            backend_manager.fillClust(shared_data.k_value); 

            double iterationSSE = backend_manager.calcIterationSSE(shared_data.k_value);
            all_iteration_sse.push_back(iterationSSE);
            
            // ---- convergence check 
            hasConverged = backend_manager.checkConvergance(shared_data ,all_iteration_sse, current_iteration, iterationSSE);
            current_iteration++; 
            
        }

        if(printRun == true)
        {
            io_manager.printIterationSSE(all_iteration_sse); 
        }
    }


    // - - - - - same as startRun but with the logging 
    void startLogRun( bool logRun, std::ofstream& path_to_output)
    {  
        int current_iteration = 0; 
        std::vector<double> all_iteration_sse; 
        bool hasConverged = false; 

        while (current_iteration < shared_data.max_iterations && !hasConverged)
        {
            setNextCentroid(); 
            backend_manager.fillClust(shared_data.k_value); 
            double iterationSSE = backend_manager.calcIterationSSE(shared_data.k_value);
            all_iteration_sse.push_back(iterationSSE);
            
            // ---- convergence check 
            hasConverged = backend_manager.checkConvergance(shared_data, all_iteration_sse, current_iteration, iterationSSE);
            current_iteration++; 
            
        }
        
        if(logRun == true)
        {
            io_manager.printIterationSSE(all_iteration_sse); 
            io_manager.logIterationSSE(all_iteration_sse, path_to_output);
        }
    }


    public :
    // ================================================= User API

   
    // - - - - - - runs the algorithm 
    void runAlg(const std::string& file_path, bool normalize_data)
    {
        // -------- configure kmeans; start run; reset  cluster_list; reset kmeans for new run

        setKmeans(true, normalize_data, file_path); // current run 0 -> 1

        bool printAllRuns = false; 
        if(shared_data.total_runs <= 10) {printAllRuns = true; }

        int currentQuarter = 1; 

        auto start = std::chrono::high_resolution_clock::now();
        while (shared_data.current_run <= shared_data.total_runs) // Current run starts at 1 end at total_runs
        {    
            printRunDecision(printAllRuns, currentQuarter); 

            setKmeans(false);
        
            shared_data.current_run++; 
        } 
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> seconds_taken = end - start;

        // print best run & it's iteration SSE 
        std::cout << "\nRun "<< shared_data.best_run_indx 
                  << " is best run :: iteration SSE\t" << shared_data.best_run_iter_sse <<std::endl; 
        std::cout << "total time taken for algorithm :: " << seconds_taken.count() << " s" << std::endl;
    }
   

    // - - - - - - runs algorithm and logs
    void runAndLogAlg
    (const std::string& path_to_data, const std::string& path_to_output, bool normalize_data)
    {

        setKmeans(true, normalize_data, path_to_data); // current run 0 -> 1

        bool logAllRuns = false; 
        if(shared_data.total_runs <= 10) {logAllRuns = true; }

        
        std::ofstream log_to_Output(path_to_output); 
                
        if(!log_to_Output.is_open())
        {
            std::string msg = "ERROR :: file is not opening"; 
            io_manager.consoleOutput(msg);
            
            msg = "INFO  :: file path = " + path_to_data;
            io_manager.consoleOutput(msg); 

            exit(EXIT_FAILURE);
        }

        int currentQuarter = 1; 
        
        auto start = std::chrono::high_resolution_clock::now();
        while (shared_data.current_run <= shared_data.total_runs) // Current run starts at 1 end at total_runs
        {    
            logRunDecision(logAllRuns, currentQuarter, log_to_Output); 

            setKmeans(false);
        
            shared_data.current_run++; 
        } 
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> seconds_taken = end - start;

        // ---- formatting to keep it consistent with console output
        std::ostringstream oss;

        oss << "\nRun " << shared_data.best_run_indx
        << " is best run :: iteration SSE\t"
        << std::fixed << std::setprecision(4) << shared_data.best_run_iter_sse
        << "\n";

        std::string outputMessage = oss.str();
        std::string timerOutput = "total time taken for algorithm :: " + std::to_string(seconds_taken.count()) + " s"; 
        
        // ---- outputs to console then to file
        io_manager.consoleOutput(outputMessage);
        io_manager.fileOutput(log_to_Output, outputMessage);
        io_manager.fileOutput(log_to_Output, timerOutput); 

        log_to_Output.close(); 
        
    }


    // ==================================================== constructor 
    k_means(int k, int runs, int max_iterations, double convergence)
    : shared_data(k, runs, max_iterations, convergence),
      backend_manager(shared_data),
      io_manager()
    {
    }

};





#endif