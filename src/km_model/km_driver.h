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
#include "./core/algorithm_backend.h"
#include "./core/model_state.h"

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
/*
    h_io is the input/output class, it holds methods that do input/output
*/
template <class T>
class algorithm_io
{

    public:
    // ============================================= basic methods that printer to console 

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
    
    // ============================================= basic methods that log to file

    // - - - - - prints iteration SSE to an output file 
    void logIterationSSE(const std::vector<double>& sse_values, std::ofstream& path_to_output)
    {
        for(int i =0; i < sse_values.size(); i++)
        {
            path_to_output << "iteration " << (i+1) << " SSE :: " << sse_values.at(i) <<std::endl; 
        }
    } 

    // - - - - - - simple log output
    void fileOutput(std::ofstream& file, std::string& output)
    {
        file << output << std::endl; 
    }



    h_io(){}
};


// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
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

    algorithm_io<T> io_manager;
    algorithm_backend<T> backend_manager; 
    model_state<T> current_state; 

    // ============================================== startRun helper method


    // - - - - - extracts partition from current clustering
    std::vector<int> labelClusterFeatures()
    {
        std::vector<int> finalDatasetClustering(current_state.data_set.size(), -1);
        
        // ------ go through each cluster
        for(int label = 0; label < current_state.k_value; label++)
        {
            clust<T>& currClust = cluster_list[k];
            std::vector<int>& clustIndicies = currClust.getAssignedData_ref();
            
            // -------- assign cluster label to each point in current cluster
            for(int i = 0; i < clustIndicies.size(); i++)
            {
                int currFeatureIndx = indices.at(i);
                finalDatasetClustering[currFeatureIndx] = label;
            }
        }
        
        return finalDatasetClustering;
    }


    // - - - - - calculates Rand Index between two partitions | good for balanced datasets - no disproportionate FN,FP,TP,TN
    //&          (truePositives + trueNegatives) / total pairs
    double calculateRandIndex(std::vector<int>& datasetLabels, std::vector<int>& groundTruth)
    {
        int totalPoints = datasetLabels.size();
        int truePositives = 0;  // same final clust in preditction & ground truth
        int trueNegatives = 0;  // different final clust in preditction & ground truth
        
        // ------ compare all pairs
        for(int x1 = 0; x1 < totalPoints; x1++)
        {
            for(int x2 = x1 + 1; x2 < totalPoints; x2++)
            {
                bool sameInPrediction = (datasetLabels[x1] == datasetLabels[x2]);
                bool sameInGroundTruth = (groundTruth[x1] == groundTruth[x2]);
                
                // correctly predicted two points in the same cluster
                if(sameInGroundTruth && sameInPrediction)
                    { truePositives++;}

                // correctly predicted two points in different clusters
                else if(!sameInGroundTruth && !sameInPrediction)
                    { trueNegatives++; }
            }
        }
        
        int totalPairs = (totalPoints * (totalPoints - 1)) / 2;
        double finalScore = (double)((truePositives + trueNegatives) / totalPairs);

        return finalScore;
    }


    // - - - - - calculates Jaccard Index between two partitions | good for disproportionate datasets - far more TN than the rest
    //&          true positives / (true positives + false negative + false positives)
    double calculateJaccardIndex(std::vector<int>& datasetLabels, std::vector<int>& groundTruth)
    {
        int totalPoints = datasetLabels.size();
        int truePositives = 0;  // same final clust in preditction & ground truth
        int falsePositives = 0;  // same final clust in preditction but different in ground truth
        int flaseNegatives = 0;  // different final clust in preditction but same in  ground truth
        
        // ------ compare all pairs
        for(int x1 = 0; x1 < totalPoints; x1++)
        {
            for(int x2 = x1 + 1; x2 < totalPoints; x2++)
            {
                bool sameInPrediction = (datasetLabels[x1] == datasetLabels[x2]);
                bool sameInGroundTruth = (groundTruth[x1] == groundTruth[x2]);
                
                // correctly predicted two points in same cluster
                if(sameInPrediction && sameInGroundTruth)
                    { truePositives++; }

                // incorrectly predected two points in the same cluster
                else if(sameInPrediction && !sameInGroundTruth)
                    { falsePositives++; }

                // 
                else if(!sameInPrediction && sameInGroundTruth)
                    { flaseNegatives++;}
            }
        }

        // does not account for true negatives | more strict but better for disproportionate sets
        //~ disproportinate where there may be 80 true positives but only 5 true negatives
        int totalPairs = truePositives + falsePositives + flaseNegatives; 
        double finalScore = (double)(truePositives / totalPairs); 
        
        return finalScore;
    }
     


    // - - - - - - finds and sets cluster's mean centroids
    void setNextCentroid()
    {
        // ------ reset clusters with new centroid
        for(int i =0; i < current_state.k_value; i++)
        {   
            
            clust<T>& currClust = current_state.cluster_list.at(i); 
            
            // store the mean data vector | call clust::genMeanFeatVector
            std::vector<T> meanCentroid_temp = currClust.genMeanFeatVector(current_state.data_set); 
            std::string id_temp = "Mean Centroid " + std::to_string(i); 

            dataPoint<T> newPoint_temp(meanCentroid_temp, 0.0, id_temp); 

            // reset cluster and assign next centroid 
            currClust.resetCluster(); 

            currClust.assignCentroid(newPoint_temp); 
        }
    }  
   
    // ============================================== log and regular run methods

    // - - - - - same as startRun but with the logging 
    void startLogRun( bool logRun, std::ofstream& path_to_output)
    {  
        int current_iteration = 0; 
        std::vector<double> all_iteration_sse; 
        bool hasConverged = false; 

        while (current_iteration < current_state.max_iterations && !hasConverged)
        {
            setNextCentroid(); 
            backend_manager.initClust/(current_state.k_value); 
            double iterationSSE = backend_manager.calcIterationSSE(current_state.k_value);
            all_iteration_sse.push_back(iterationSSE);
            
            // ---- convergence check 
            hasConverged = backend_manager.checkConvergance(current_state, all_iteration_sse, current_iteration, iterationSSE);
            current_iteration++; 
            
        }
        
        double convergedSSE = all_iteration_sse.at(all_iteration_sse.size()-1); 
        double ShiloetteScore = backend_manager.genShiloetteScore(current_state.k_value);
        double CHI_score = backend_manager.genCHI(convergedSSE, current_state.k_value);  

        current_state.checkBetterChi(CHI_score);
        current_state.checkBetterShil(ShiloetteScore);

        if(logRun == true)
        {
            std::string msg =  "\nthe Shiloette Score for this run :: " + std::to_string(ShiloetteScore); 
            msg.append( "\nthe CHI index score for this run is :: " + std::to_string(CHI_score) + "\n");

            io_manager.printIterationSSE(all_iteration_sse); 
            io_manager.logIterationSSE(all_iteration_sse, path_to_output);
            
            io_manager.consoleOutput(msg); 
            io_manager.fileOutput(path_to_output, msg); 
        }
    }

    // - - - - - run without logging 
    void startRun(bool printRun)
    {  
        int current_iteration = 0; 
        std::vector<double> all_iteration_sse; 
        bool hasConverged = false; 


        while (current_iteration < current_state.max_iterations && !hasConverged)
        {
            setNextCentroid(); 
            backend_manager.initClust(current_state.k_value); 

            double iterationSSE = backend_manager.calcIterationSSE(current_state.k_value);
            all_iteration_sse.push_back(iterationSSE);
            
            // ---- convergence check 
            hasConverged = backend_manager.checkConvergance(current_state ,all_iteration_sse, current_iteration, iterationSSE);
            current_iteration++; 
            
        }

        double convergedSSE = all_iteration_sse.at(all_iteration_sse.size()-1); 
        double ShiloetteScore = backend_manager.genShiloetteScore(current_state.k_value);
        double CHI_score = backend_manager.genCHI(convergedSSE, current_state.k_value);  

        current_state.checkBetterChi(CHI_score);
        current_state.checkBetterShil(ShiloetteScore);

        std::vector<int> runFinalClusterings = labelClusterFeatures(); 
        double randIndx = calculateRandIndex(runFinalClusterings, current_state.target_clusters); 
        double jaccardIndx = calculateJaccardIndex(runFinalClusterings, current_state.target_clusters);

        current_state.checkBetterJaccard(jaccardIndx); 
        current_state.checkBetterRandIndx(randIndx);

        if(printRun == true)
        {
            std::string msg =  "\nthe Shiloette Score for this run :: " + std::to_string(ShiloetteScore); 
            msg.append( "\nthe CHI index score for this run is :: " + std::to_string(CHI_score));
            msg.append( "\nthe rand index for this run is :: " + std::to_string(randIndx));
            msg.append( "\nthe jaccard index for this run is :: " + std::to_string(jaccardIndx)); 

            io_manager.printIterationSSE(all_iteration_sse); 
            io_manager.consoleOutput(msg);
        }
    }

    // ============================================ print decision methods

    // - - - - - decides what to print / log
    void logRunDecision(bool logAllRuns, int& currentQuarter, std::ofstream& outFile)
    {
        std::string mainOutput = "Run " + std::to_string(current_state.current_run); 

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
            if (current_state.total_runs % 2 == 0)
                {Qinterval = (current_state.total_runs) / 4;}
    
            else 
                {Qinterval = (current_state.total_runs -3) /4; }
        
            // varrifies that my printing interval isn't too small
            if(Qinterval < 2) 
                {Qinterval == 2; }


            // prints 1st run | prints last run | prints quarter runs
            if(current_state.current_run == 1 )
            { 
                mainOutput.append(" | first Run\n-------------\n");

                io_manager.consoleOutput(mainOutput); 
                io_manager.fileOutput(outFile, mainOutput); 
                printCurrentRun = true; 
            }
            
            else if(current_state.current_run == current_state.total_runs)
            {
                mainOutput.append(" | last Run\n-------------\n");
                
                io_manager.consoleOutput(mainOutput); 
                io_manager.fileOutput(outFile, mainOutput); 
                printCurrentRun = true; 
            }

            else if(current_state.current_run % Qinterval == 0)
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

    // - - - - - - backend for print | logic for when to print and what to print 
    void printRunDecision(bool printAllRuns, int& currentQuarter)
    {
          // prints all runs for small ranges of run | total_run <= 10  
            std::string mainOutput = "\nRun " + std::to_string(current_state.current_run);    
            
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
                if (current_state.total_runs % 2 == 0)
                    {Qinterval = (current_state.total_runs) / 4;}
                
                else 
                    {Qinterval = (current_state.total_runs -3) /4; }
                
                // varrifies that my printing interval isn't too small
                if(Qinterval < 2) 
                    {Qinterval == 2; }

                // prints 1st run | prints last run | prints quarter runs
                if(current_state.current_run == 1 )
                { 
                    mainOutput.append("\n--------------\n");
                    io_manager.consoleOutput(mainOutput); 
                    printCurrentRun = true; 
                }

                else if(current_state.current_run == current_state.total_runs)
                {
                    mainOutput.append(" | last Run\n-------------\n");
                    io_manager.consoleOutput(mainOutput); 
                    printCurrentRun = true; 
                }

                else if(current_state.current_run % Qinterval == 0)
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

    // ============================================== init & reset alg for next run
    
    // - - - - - sets the algorithm | initial set and reset between runs 
    void setKmeans(bool isInit, bool normalize = false, const std::string& file_path = "")
    {
        for(int i =0; i < current_state.k_value; i++)
        {
            clust<T> clust_temp; 
            clust_temp.setID(i);  
            current_state.cluster_list.push_back(std::move(clust_temp));
        }

        if(isInit)
        {
            backend_manager.loadDataSet(file_path);
        }
        //~ ------ initialization strategy goes here 
    
         backend_manager.init_forging(current_state.k_value); 
    
        // ------ currentRun = 0 is initial run 
        //        decide to normalize or not then increase currentRun from 0 to 1
        if(isInit)
        {
            if(normalize == true)
                { backend_manager.calcNormalizedData(); }
           
            current_state.current_run++;    
        }
    }


    public : 

    // ============================================== API driver methods
    // - - - - - - runs the algorithm 
    void runAlg(const std::string& file_path, bool normalize_data)
    {
        // -------- configure kmeans; start run; reset  cluster_list; reset kmeans for new run

        setKmeans(true, normalize_data, file_path); // current run 0 -> 1

        bool printAllRuns = false; 
        if(current_state.total_runs <= 10) {printAllRuns = true; }

        int currentQuarter = 1; 

        //& - - - - - - - - start of timer & main algorithm 

        auto start = std::chrono::high_resolution_clock::now();
        while (current_state.current_run <= current_state.total_runs) // Current run starts at 1 end at total_runs
        {    
            printRunDecision(printAllRuns, currentQuarter); 

            setKmeans(false);
        
            current_state.currentRun_increase(); 
        } 
        auto end = std::chrono::high_resolution_clock::now();

        //& - - - - - - - - end of timer and main algorithm 

        std::chrono::duration<double> seconds_taken = end - start;

        // print best run & it's iteration SSE 
        std::cout << "-----------------------\n"
        << "the best run iteration SSE ::\t"        << current_state.best_run_iter_sse << std::endl
        << "the best CHI score ::\t\t"              << current_state.best_chi_score << std::endl
        << "the best shiloette score ::\t"          << current_state.best_shiloette << std::endl;

        std::cout << "the best rand index score ::\t"     << current_state.best_jaccard_index_score << std::endl
                  << "the best jaccard index score ::\t"  << current_state.best_rand_indx_score << std::endl;

        std::cout << "total time taken for algorithm :: "   << seconds_taken.count() << "s" << std::endl;
        
    }
   
    // - - - - - - runs algorithm and logs
    void runAndLogAlg
    (const std::string& path_to_data, const std::string& path_to_output, bool normalize_data)
    {

        setKmeans(true, normalize_data, path_to_data); // current run 0 -> 1

        bool logAllRuns = false; 
        if(current_state.total_runs <= 10) {logAllRuns = true; }

        //& - - - - - - load and test output files
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
        
        //& - - - - - - start of the timer and main algorithm 
        auto start = std::chrono::high_resolution_clock::now();
        while (current_state.current_run <= current_state.total_runs) // Current run starts at 1 end at total_runs
        {    
            logRunDecision(logAllRuns, currentQuarter, log_to_Output); 

            setKmeans(false);
        
            current_state.currentRun_increase(); 
        } 
        
        auto end = std::chrono::high_resolution_clock::now();
        //& - - - - - end of the timer and main algorithm 

        std::chrono::duration<double> seconds_taken = end - start;

        //& - - - - - output format 
        // ---- formatting to keep it consistent with console output
        std::ostringstream oss;

          
        oss << "-------------------------\n" 
        <<"the best run iteration SSE ::\t" << std::fixed << std::setprecision(4) << current_state.best_run_iter_sse << "\n"
        << "the best CHI score ::\t\t"            << current_state.best_chi_score << "\n"
        << "the best shiloette score ::\t"      << current_state.best_shiloette << "\n"; 

                                                
        std::string outputMessage = oss.str();
        std::string timerOutput = "total time taken for algorithm :: " + std::to_string(seconds_taken.count()) + " s"; 
        
        //& - - - - - outputs
        // ---- outputs to console then to file
        io_manager.consoleOutput(outputMessage);
        io_manager.fileOutput(log_to_Output, outputMessage);
        io_manager.fileOutput(log_to_Output, timerOutput); 

        log_to_Output.close(); 
        
    }

    // ==================================================== constructor 
    k_means(int k, int runs, int max_iterations, double convergence)
    : current_state(k, runs, max_iterations, convergence),
      backend_manager(current_state),
      io_manager()
    {
    }

};

#endif