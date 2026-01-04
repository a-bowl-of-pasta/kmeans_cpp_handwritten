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
    model_state<T> shared_data; 

    // ================================================= core methods

    // - - - - - sets the algorithm | initial set and reset between runs 
    void setKmeans(bool isInit, bool normalize = false, const std::string& file_path = "")
    {
        for(int i =0; i < shared_data.k_value; i++)
        {
            clust<T> clust_temp; 
            clust_temp.setID(i);  
            shared_data.cluster_list.push_back(std::move(clust_temp));
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
            
            // store the mean data vector | call clust::genMeanFeatVector
            std::vector<T> meanCentroid_temp = currClust.genMeanFeatVector(shared_data.data_set); 
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

        double convergedSSE = all_iteration_sse.at(all_iteration_sse.size()-1); 
        double ShiloetteScore = backend_manager.genShiloetteScore(shared_data.k_value);
        double CHI_score = backend_manager.genCHI(convergedSSE, shared_data.k_value);  

        shared_data.checkBetterChi(CHI_score);
        shared_data.checkBetterShil(ShiloetteScore);

        if(printRun == true)
        {
            std::string msg =  "\nthe Shiloette Score for this run :: " + std::to_string(ShiloetteScore); 
            msg.append( "\nthe CHI index score for this run is :: " + std::to_string(CHI_score));

            io_manager.printIterationSSE(all_iteration_sse); 
            io_manager.consoleOutput(msg);
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
        
        double convergedSSE = all_iteration_sse.at(all_iteration_sse.size()-1); 
        double ShiloetteScore = backend_manager.genShiloetteScore(shared_data.k_value);
        double CHI_score = backend_manager.genCHI(convergedSSE, shared_data.k_value);  

        shared_data.checkBetterChi(CHI_score);
        shared_data.checkBetterShil(ShiloetteScore);

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
        std::cout << "-----------------------\n"
        << "the best run iteration SSE ::\t"    << shared_data.best_run_iter_sse <<std::endl
        << "the best CHI score ::\t\t"                        << shared_data.best_chi_score << std::endl
        << "the best shiloette score ::\t"                  << shared_data.best_shiloette << std::endl;
        std::cout << "total time taken for algorithm :: "   << seconds_taken.count() << "s" << std::endl;
        
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

          
        oss << "-------------------------\n" 
        <<"the best run iteration SSE ::\t" << std::fixed << std::setprecision(4) << shared_data.best_run_iter_sse << "\n"
        << "the best CHI score ::\t\t"            << shared_data.best_chi_score << "\n"
        << "the best shiloette score ::\t"      << shared_data.best_shiloette << "\n"; 

                                                
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