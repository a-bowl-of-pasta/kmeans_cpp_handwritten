
#include "kmeans.h"

/*
void first_centroid()
{
        // K is larger than the size of the clusterData
        if(clusterData.size() < numClust)
        {
            std::cout << "ERROR :: not enough data for " << k << " clusters" << std::endl; 
            std::cout << "INFO  :: data Size = " << clusterData.size() << " cluster amount = " << k << std::endl; 
            return; 
        }
        

        int* randIndex = new int[numClust]; // used to make sure there are no duplicates 
        int centroid_index =0; 

        // run until all centroids are generated
        while (centroid_index < numClust)
        {

            int nextRand = rand() % clusterData.size();
            bool dup = false; 

            // check previous random nums to make sure nextRand is not a duplicate
            for(int i =0; i < centroid_index; i++)
            {
                if(randIndex[i] == nextRand)
                {
                    dup = true; 
                    break; 
                }                
            }

            // if nextRand is a duplicate, do not add point | if nextRand is not dup, add point
            if(dup == false)
            {            
                // add nextRand to array holding previous random numbers
                randIndex[centroid_index] = nextRand; 
                
                // overwrite old data ID with centroid ID
                std::string centroid_id = "centroid " + std::to_string(centroid_index + 1); 
                clusterData.at(nextRand).setID(centroid_id); 

                // add point from clusterData to centroid container
                centroids[centroid_index] = clusterData.at(nextRand);
                centroid_index++; 
            }
    }
    delete[] randIndex; 
}


int main(int argc, char* argv[])
{

    // ===== arg variables 
    std::string filename; //argv[1]
    int k_val;           // argv[2]
    int iterations;     // argv[3]
    double convergence;// argv[4]
    int num_of_runs;  // argv[5] 


    // ===== makes sure variables are valid 
    try
    {
        filename = argv[1]; 
        k_val = std::stoi(argv[2]); 
        iterations = std::stoi(argv[3]); 
        convergence = std::stod(argv[4]);
        num_of_runs = std::stoi(argv[5]); 

        if (k_val <= 1)
        {
            std::cout << "ERROR :: K_value <" <<k_val<<"> should be larger than 1"<< std::endl;
            exit(EXIT_FAILURE);  
        }

    }catch(std::exception e)
    {
        std::cout << "ERROR :: invalid arguments | should be - <string> <int> <int> <double> <int>" << std::endl; 
        exit(EXIT_FAILURE);  
         
    }
    
    // ==========  gen path name | seed rand | create cluster obj | gen data ID numbers 
    // - - - - - - - - - - - - - - - - - - - - - - - -
    // use this path when running through the bat File
    std::string path = "./code_and_files/data_sets/" + filename; 
    
    // use this path when running through the cpp File 
    // std::string path = "../data_sets/" + filename;
    // - - - - - - - - - - - - - - - - - - - - - - - 
    
    srand(time(0));

    // ======== populating the cluster with the data
    try
    {


        
        // set and display centroids
        cluster.setCentroids(); 
        cluster.displayCentroids(); 

        // - - - - - - - - - - - - - - - - - - - - - - - - 
        //use this if running through cpp file
        //std::string log_path = "../../output_files/o_" + filename;
        
        //use this if running through bat file
        std::string log_path = "./output_files/o_"+filename; 
        // - - - - - - - - - - - - - - - - - - - - - - - -
        
        std::ofstream outputFile(log_path);

        if(!outputFile.is_open())
        {
            std::cout << "ERROR :: outputFile is not opened"<< std::endl; 
            std::cout << "INFO  :: output path = " << log_path << std::endl; 
            exit(EXIT_FAILURE);
        }
        
        cluster.logCentroids(outputFile);
        

        outputFile.close(); 
    }
    catch(std::exception e) 
    {
        std::cout << "ERROR :: file " + path + " not found" << std::endl;  
        exit(EXIT_FAILURE);  
    }
    


return 0; 
}

*/