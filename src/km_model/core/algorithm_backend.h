 #ifndef ALG_BACKND
#define ALG_BACKND

/*
    the h_backend class holds the backend logic for the k_means class
    it is for methods that performs calculations and data processing
    
*/
template <class T>
class algorithm_backend
{

    std::vector<dataPoint<T>>& data_set;
    std::vector<dataPoint<T>>& target_values;   
    std::vector<T>& max_data_vector; 
    std::vector<T>& min_data_vector; 
    std::vector<clust<T>>& cluster_list; 

    void createDataSetNode(const std::string& line, int& id, bool& firstRun)
    {           
        // read rows element by element
        std::istringstream lineStream(line);
        std::string features; 
        std::vector<double> featureVector_temp;

        int currFeature =0; 
        while(getline(lineStream, features, ' '))
        {
            // ---- skip empty elements 
            if (!features.empty())
            {
                double featToDec = std::stod(features); 
               
                // ----- first run populates min/max vectors, other runs find min/max
                if(firstRun == true)
                {
                    maxDataVector.push_back(featToDec);
                    minDataVector.push_back(featToDec);
                }
                else
                {
                    // swap attributes if elm in vector is smaller
                    if(maxDataVector.at(currFeature) < featToDec)
                    {
                        maxDataVector.at(currFeature) = featToDec; 
                    }
                    
                    // swap attributes if elm in vector is greater
                    if(minDataVector.at(currFeature) > featToDec)
                    {
                        minDataVector.at(currFeature) = featToDec; 
                    }
                }
                featureVector_temp.push_back(featToDec);
                currFeature++; 
            }
        }

        // ---- skip empty vectors  
        if (!featureVector_temp.empty())
        {
            double targetValue = featureVector_temp.pop_back();

            std::string point_ID_temp = "dataPoint " + std::to_string(id); 
            data_set.push_back(dataPoint<T>(featureVector_temp, 0.0, point_ID_temp));
            target_values.push_back(targetValue); 
            id++; 

        }              
        
        if(firstRun == true) {firstRun = false; }

    }
    
    
    bool checkUnique(int currClustIndx, int randCentroidIndx)
    {
        const std::string& dataPointID = data_set.at(randCentroidIndx).getDataID(); 
        bool unique = true; 

        for(int i = 0; i < currClustIndx; i++)
        {
            dataPoint<T>& prevCentroids = cluster_list.at(i).getCentroid_ref(); 
                 
            if(dataPointID == prevCentroids.getDataID())
            {
                unique = false; 
                break; 
            }
        }

        return unique; 
    }
   

    public: 

    // ==================================================== distance calculations

    //  - - - - - - - - Finds the euclidean distance of two points (x1, x2)
    //&                 (X1_0 - X2_0)^2 + (X1_1 - X2_1)^2 + ..... + (X1_n - X2_n)^2
    double sqr_euclid_dist(std::vector<T>& x1_feature_vector, std::vector<T>& x2_feature_vector)
    {
        double finalDistance = 0.0; 
           
        // - - - - loop through x2 features 
        for(int i =0; i < x1_features.size(); i++)
        {
            // get the features out of the vector 
            T x2_feature = x2_features.at(i); 
            T x1_feature = x1_features.at(i); 

            // residual = x1 - x2 
            double curr_sqr_residual = x1_feature - x2_feature; 
           
            // sqr_residual = residual^2
            curr_sqr_residual = curr_sqr_residual * curr_sqr_residual; 

            // finalDist = sum( all sqr_residuals )
            finalDistance += curr_sqr_residual; 
        }

        return finalDistance; 
    }    

    // - - - - - - finds regular euclidean distance
    //&            sqrt( (X1_0 - X2_0)^2 + (X1_1 - X2_1)^2 + ..... + (X1_n - X2_n)^2 )
    double euclidean_distance(std::vector<T>& x1_feature_vector, std::vector<T&> x2_feature_vector)
    {
        double temp = sqr_euclid_dist(x1_feature_vector, x2_feature_vector);
        return sqrt(temp); 
    }

    // - - - - - - - finds the manhattan distance of two points (x1, x2)
    //&              |X1_0 - X2_0| + |X1_1 - X2_1| + ..... + |X1_n - X2_n|
    double manhattan_distance(std::vector<T>& x1_feature_vector, std::vector<T>& x2_feature_vector)
    {
        double finalDistance = 0.0; 
           
        // - - - - loop through x2 features 
        for(int i =0; i < x1_features.size(); i++)
        {
            // get the features out of the vector 
            T x2_feature = x2_features.at(i); 
            T x1_feature = x1_features.at(i); 

            // residual = |x1 - x2|
            double curr_abs_residual = std::abs(x1_feature - x2_feature); 
        
            // finalDist = sum( all residuals )
            finalDistance += curr_abs_residual; 
        }

        return finalDistance; 
    }    


    // ============================================ processing 

    // - - - - - - -  -  assign data from  data_set to cluster 
    // ! - - - - - - - - - - - - - - - - - was nemed fillClust
    void initClust(int k_value) 
    {
        int currentPoint = 0;

        // ----- loops through data set to find closest centroid per dataPoint
        //       Clust saves the dataPoint index from  data_set, not the dataPoint itself
        while(currentPoint < data_set.size())
        {
            int bestFitIndex_temp = 0; 
            double currBestFit_temp = std::numeric_limits<double>::max();    

            // - - - finds distance between each centroid and point
            for(int i = 0; i < k_value; i ++)
            {
                dataPoint<T>& currCentroid_temp = cluster_list.at(i).getCentroid_ref();
                double distFromCent_temp = sqr_euclid_dist(data_set.at(currentPoint), currCentroid_temp);
                
                if(distFromCent_temp < currBestFit_temp )
                {
                    bestFitIndex_temp = i; 
                }
               
            }

            cluster_list.at(bestFitIndex_temp).assignData(currentPoint); 
            currentPoint++; 
        } 
        
    }    


    // - - - - finds the iteration SSE
    double calcIterationSSE(int k_value)
    {
        double totalIterationSSE = 0.0;
        for(int i =0; i < k_value; i++)
        {
            cluster_list.at(i).genSSE(data_set); 
            totalIterationSSE += currClust.getClassLevelSSE(); 
        }
        return totalIterationSSE; 
    }
   

    // - - - - - - CH = (SSB / (k - 1)) / (SSW / (n - k))
    double genCHI(double runSSE, int k_value)
    {
        
        double SSB_value = 0.0; 
        std::vector<T> meanDataVec =  (); 
        for(int i = 0; i < k_value; i++)
        {
            clust<T>& currClust = cluster_list.at(i); 
            dataPoint<T> currCent = currClust.getCentroid(); 

            // gets the squared euclidean distance | (centroid - meanVec)^2
            double pt1 = currCent.calcSquaredEuclidDist(meanDataVec); 

            SSB_value += currClust.getSize() * pt1; 
        }

        double pt1 = SSB_value / (k_value - 1); 
        double pt2 = runSSE / (data_set.size() - k_value); 

        return pt1 / pt2; 
    }


    // - - - - - - - the total Silhouette score for the current run
    double genShiloetteScore(int k_value)
    {
        double runSScore = 0.0 ;


        for(int i = 0; i < k_value; i++)
        {    
            clust<T>& currClust = cluster_list.at(i); 
            currClust.genClustSilhouetteScore(data_set, cluster_list, k_value);
            runSScore += currClust.getSScoreContribution(); 
        }
        return (runSScore / data_set.size()); 
    }


    // - - - - - - reads input file, 
    void loadDataSet( const std::string&  data_file, )
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

    // - - - - random data point init
    void init_forging(int k_value)
    {
        int currentCluster = 0; 

        // ---- better random numbers
        std::random_device rand_seed; 
        std::mt19937 PRNG(rand_seed());
        std::uniform_int_distribution<int> numRange(0, data_set.size() -1); 
        
        // -------- runs until all K clusters have centroids
        while (currentCluster < k_value)
        {
            int centroidIndx = numRange(PRNG); 

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

        initClust(k_value); 
    }


    // - - - - creates a new node (dataPoint) for the dataSet vector | not used outside this class
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
   

    // - - - -  data normalization
    void calcNormalizedData()
    {

        // ------ goes through each feature vector
        for(int currFeatVectIndx =0; currFeatVectIndx < data_set.size(); currFeatVectIndx++)
        {
            std::vector<T>& currFeatVect_temp = data_set.at(currFeatVectIndx).getFeatureVector_ref(); 
            
            // ----- goes through each elm in feature vector
            for(int featureIndx = 0; featureIndx < currFeatVect_temp.size(); featureIndx++ )
            {
                double currFeature_temp = currFeatVect_temp.at(featureIndx); 
                double maxVal_temp = maxDataVector.at(featureIndx); 
                double minVal_temp = minDataVector.at(featureIndx); 
                
                // x' = x - min(x) / max(x) - min(x)
                double normalizedFeature = 0.0; 
                if((maxVal_temp - minVal_temp) != 0)
                {
                    normalizedFeature = (currFeature_temp - minVal_temp) / (maxVal_temp - minVal_temp);
                }

                currFeatVect_temp.at(featureIndx) = normalizedFeature; 
            }
            data_set.at(currFeatVectIndx).setFeatureVector(currFeatVect_temp);
        }
    }

    // - - - - random partition init
    void init_randomPartition(int k_value)
    {
        // picks random cluster form cluster list
        std::random_device rand_seed;
        std::mt19937 PRNG(rand_seed());
        std::uniform_int_distribution range(0, k_value-1);
        
        // ---- assign data to rand clust
        for(int i = 0 ; i < data_set.size(); i++)
        {
            int randNum = range(PRNG);
            
            cluster_list.at(randNum).assignData(i);
        }   

        // ---- find mean featVec for each clust then assign as centroid
        for(int k = 0; k < k_value; k++)
        {

            std::vector<T> meanData_temp = clust_list.at(k).genMeanFeatVector(data_set);  
            
            std::string id_temp = "Mean Centroid " + std::to_string(k); 
            dataPoint<T> newPoint_temp(meanData_temp, 0.0, id_temp);

            clust_list.at(k).assignCentroid(newPoint_temp);
        }

        // reassign data based off of the initial centroid
        initClust(k_value); 
    }


    algorithm_backend(dataBucket<T>& shared_data)
    : data_set(shared_data.data_set), 
      max_data_vector(shared_data.maxDataVector), 
      min_data_vector(shared_data.minDataVector), 
      cluster_list(shared_data.cluster_list),
      target_values(shared_data.target_values)
      {}

    algorithm_backend()
    : data_set(), 
      max_data_vector(), 
      min_data_vector(), 
      cluster_list(),
      target_values() 
      {}
};

#endif