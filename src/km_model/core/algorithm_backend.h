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
  
    // - - - - finds the iteration SSE
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
   
     // - - - - - - gets me the mean vector for the whole dataSet
     std::vector<T> dataSetMeanVector()
    {
        dataPoint<T>& temp = data_set.at(0); 
        int numOfFeatures = temp.getDataVector().size(); 
    
        std::vector<double> meanFeatureVector(numOfFeatures, 0.0); 
    
        int sizeOfData = data_set.size();  

        // ------ go through each data vector
        for(int dataVector = 0; dataVector < sizeOfData; dataVector++)
        {
            std::vector<T>& elms = data_set.at(dataVector).getDataVectorByRef(); 
            
            // -------- calculate the sum of each feature
           for(int element = 0; element < elms.size(); element++  )
            {
                meanFeatureVector.at(element) += elms.at(element); 
            }

        }

        // ------- divide each feature's sum by total dataPoints
        // ------- this creates a vector of means
        for(int i =0; i < meanFeatureVector.size(); i++)
        {
            meanFeatureVector.at(i) = meanFeatureVector.at(i) / sizeOfData; 
        }

        // ---- return new centroid's dataVector
        return meanFeatureVector; 

    }

    // - - - - - SSB = sum(total_clust_dataPoints * ||centroid_dataVec - dataset_meanVec||^2) 
    double findSSB(int k_value)
    {
        double SSB_value = 0.0; 
        std::vector<T> meanDataVec = dataSetMeanVector(); 
        for(int i = 0; i < k_value; i++)
        {
            clust<T>& currClust = cluster_list.at(i); 
            dataPoint<T> currCent = currClust.getCentroid(); 

            // gets the squared euclidean distance | (centroid - meanVec)^2
            double pt1 = currCent.calcSquaredEuclidDist(meanDataVec); 

            SSB_value += currClust.getSize() * pt1; 
            
        }
    
        return SSB_value; 
    }

    // - - - - - - CH = (SSB / (k - 1)) / (SSW / (n - k))
    double genCHI(double runSSE, int k_value)
    {
        double SSB = findSSB(k_value); 
        double pt1 = SSB / (k_value - 1); 
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

#endif