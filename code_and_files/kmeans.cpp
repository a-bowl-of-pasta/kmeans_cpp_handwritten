
#include "./header_files/kmeans.h"
#include "./header_files/clusters.h"
#include <vector>
#include <iostream>

// ================================ test with hard coded values
void inlineRuns()
{
    // ===== arg variables 
    std::string file_name = "iris_bezdek";           //argv[1]
    int k_val = 3;                           // argv[2]
    int iterations = 100;                    // argv[3]
    double convergence = 0.001;             // argv[4]
    int num_of_runs = 100;                 // argv[5] 

    std::string path_in = "../default_data_sets/" + file_name + ".txt";
    std::string path_out = "../outputs/v1_outputs/" + file_name + "_out.txt";


    k_means<double> km_manager(k_val, num_of_runs, iterations, convergence);
   
    km_manager.runAlgorithm(path_in); 
    //km_manager.runAndLogAlg(path_in, path_out); 

}

// =================================== run using args
void actualRun(int argc, char* argv[])
{
     // ===== arg variables 
    std::string file_name; //argv[1]
    int k_val;           // argv[2]
    int iterations;     // argv[3]
    double convergence;// argv[4]
    int num_of_runs;  // argv[5] 


    // ======================= check that args are correct & create k_means API thingy
    try
    { 
        file_name = argv[1]; 
    }
    catch(std::exception e)
    {
        std::cout << "ERROR :: invalid argument | file_Name should be <String>" << std::endl;
        std::cout << "INFO  :: fileName = " << file_name << std::endl;  
        exit(EXIT_FAILURE);        
    }

    std::string path_in = "../default_data_sets/" + file_name + ".txt";
    std::string path_out = "../outputs/v1_outputs/" + file_name + "_out.txt";

    k_means<double> km_manager(std::stoi(argv[5]), std::stod(argv[4]), std::stoi(argv[2]), std::stoi(argv[3]));
    km_manager.runAlgorithm(path_in); 


}

// ============================= main 
int main(int argc, char* argv[])
{

    // ---- run using args
    //actualRun(argc, argv);
    
    // ---- test using hard coded values 
    inlineRuns(); 

return 0; 
}