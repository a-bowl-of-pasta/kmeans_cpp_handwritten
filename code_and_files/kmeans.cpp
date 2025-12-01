
#include "./header_files/kmeans.h"
#include "./header_files/clusters.h"
#include <vector>
#include <iostream>

// ================================ test with hard coded values
void inlineRuns()
{
    // hard coded file names for argv[1]
    int indx_used = 9; 
    std::string current_version_out = "v3.0_outputs";

    std::string default_files[] = {"ecoli", "glass", "ionosphere", "iris_bezdek", "landsat", "letter_recognition", "segmentation", "vehicle", "wine", "yeast"}; 
    int default_Kvals[] = {8,6,2,3,6,26,7,4,3,10};
    
    // manually typing file name for argv[1]
    //std::string file_name = "testingMM";           

    // ===== arg variables 
    int k_val = default_Kvals[indx_used];     // argv[2]
    int iterations = 100;                    // argv[3]
    double convergence = 0.001;             // argv[4]
    int num_of_runs = 100;                 // argv[5] 

    // so that you can copy and paste :: default_files[indx_used]    file_name
    std::string path_in = "../default_data_sets/" + default_files[indx_used] + ".txt";
    std::string path_out = "../outputs/"+ current_version_out +"/"+ default_files[indx_used] + "_out.txt";


    k_means<double> km_manager(k_val, num_of_runs, iterations, convergence);
    //km_manager.runAlg(path_in, true); 
    km_manager.runAndLogAlg(path_in, path_out, true); 

}

// =================================== run using args
void actualRun(int argc, char* argv[])
{
     // ===== arg variables 
    std::string file_name; //argv[1]
    int k_val;            // argv[2]
    int iterations;      // argv[3]
    double convergence; // argv[4]
    int num_of_runs;   // argv[5] 


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

    std::string path_in = "../data_sets/" + file_name + ".txt";

   
    k_means<double> km_manager(std::stoi(argv[5]), std::stod(argv[4]), std::stoi(argv[2]), std::stoi(argv[3]));
    km_manager.runAlg(path_in, true); 


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

/*
    intended interactions 
    user
    |
    | - - - k_means
            |
            | - - - km_dat
            | - - - h_io   
            | - - - h_backend
                    | - - - DataPoint
                    | - - - Clust
                            |
                            | - - - DataPoint 
*/