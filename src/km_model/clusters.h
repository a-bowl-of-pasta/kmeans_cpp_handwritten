#ifndef CLUST_STUFF 
#define  CLUST_STUFF

#include <iostream>
#include <string>
#include <vector>
   


// ================================================= dataPoints structure
/*
    the dataPoint class consists of a dataPoint's
    MetaData and DataVector.

    the DataVector is the information from the DataSet
    the MetaData is the extra helpful stuff

    the template is to define what kind of data
    the DataVectors are storing 
*/
template <class T>
struct dataPoint
{
    // data ID | distance from nearest cluster | data
    std::string dataID; 
    std::vector<T> data_point;
    double distance_from_cluster; 
   
   
    // = = = = = = = = = = = = = = setters and getters 
    void setID(std::string& id) 
        { dataID = std::move(id); }
    
    void replaceDataVector(std::vector<T>& updatedVector)
        { data_point = std::move(updatedVector); }

    std::string& getDataID()
        { return dataID;}


    std::vector<T> getDataVector()
        { return data_point;}

    
    std::vector<T>& getDataVectorByRef()
        { return data_point; }    

    // = = = = = = = = = = = = = = Helpful methods

    
    // - - - - finds the squared euclidean distance between this and another point
    double calcSquaredEuclidDist(dataPoint<T>& x_2)
    {
        std::vector<T>& x_2_dataVec = x_2.getDataVectorByRef();
        double squaredDistance = 0.0; 
           
        // ---- (X1_0 - X2_0)^2 + (X1_1 - X2_1)^2 + ..... + (X1_n - X2_n)^2
        for(int i =0; i < data_point.size(); i++)
        {
            T x_2_elm = x_2_dataVec.at(i); 
            T x_1_elm = data_point.at(i); 

            double squaredDiff = x_1_elm - x_2_elm; 
            squaredDiff = squaredDiff * squaredDiff; 
            squaredDistance += squaredDiff; 
        }

        return squaredDistance; 
    }


    double calcSquaredEuclidDist(std::vector<T>& x_2_dataVec)
    {
        double squaredDistance = 0.0; 
           
        // ---- (X1_0 - X2_0)^2 + (X1_1 - X2_1)^2 + ..... + (X1_n - X2_n)^2
        for(int i =0; i < data_point.size(); i++)
        {
            T x_2_elm = x_2_dataVec.at(i); 
            T x_1_elm = data_point.at(i); 

            double squaredDiff = x_1_elm - x_2_elm; 
            squaredDiff = squaredDiff * squaredDiff; 
            squaredDistance += squaredDiff; 
        }

        return squaredDistance; 
    }


    // = = = = = = = = = = = = = = visualization methods 
    void printData()
    {
        std::cout << dataID << " : "; 
        for (int i =0; i < data_point.size(); i++)
        {
            std::cout << data_point.at(i) << "\t"; 
        }
    }



    void logData(std::ofstream& file)
    {
        file << dataID << " : "; 
        for(int i =0; i < data_point.size(); i++)
        {
            file << data_point.at(i) << "\t"; 
        }
    }



    // = = = = = = = = = = = = = = constructors 
    dataPoint(std::vector<T>& dataPoint, double distanceFromClust, std::string& id)
    {
        dataID = std::move(id); 
        data_point = std::move(dataPoint);
        distance_from_cluster = distanceFromClust; 
    }
    dataPoint()
    {
        dataID = ""; 
        data_point = std::vector<T>{};  
        distance_from_cluster = 0.0; 
    }
};


// ========================================================== kmeans class 
/*
    the clust class is in charge of the clusters
    it holds cluster specific information.

    clust_data_indicies is the data assigned to the
    cluster. copying the dataSet from Kmeans is too
    much work, so instead it just stores indicies. 

    the Template T is to be passed down to the 
    DataPoint class, so that it knows what the
    DataVectors are storing. 
*/
template <class T>
class clust
{
    
    double class_level_SSE;
    double total_SScore;
    double class_level_SScore; 
    std::string clust_id; 
    dataPoint<T> centroid; 
    std::vector<int> clust_data_indicies;


    public: 
    // = = = = = = = = = = = = = =  short & basic methods    
    void assignData(int dataSetIndex)
        { clust_data_indicies.push_back(dataSetIndex);}

        

    void assignCentroid(dataPoint<T>& cent) 
        { centroid = cent;}


    void setID(int id)
        { clust_id = "c" + std::to_string(id); }



    dataPoint<T> getCentroid()
        {return centroid; }



    double getClassLevelSSE()
        {return class_level_SSE;}


    double getSScoreContribution()
        {return total_SScore;}



    std::vector<int> getDataIndicies()
        {return clust_data_indicies; }
   

    std::string getID()
        {return clust_id;}

    
    int getSize()
        {return clust_data_indicies.size();}


    void resetCluster()
    {
        centroid = dataPoint<T>{};
        class_level_SSE = 0.0; 
        total_SScore = 0.0; 
        class_level_SScore = 0.0; 
        clust_data_indicies.clear();
    }


    // = = = = = = = = = = = = = = visualization methods
    void printCentroid()
    {
        centroid.printData(); 
        std::cout << std::endl; 
    }
    
    
    
    void logCentroids(std::ofstream& file)
    {
            centroid.logData(file); 
            file << std::endl; 
    }   
    


    void printDataPoint(const std::vector<dataPoint<T>>& dataSet, int index)
    { 
            int pos = clust_data_indicies.at(index); 
            dataPoint<T>& data = dataSet.at(pos); 
            data.printData(); 
            std::cout << std::endl; 
    }
   


    // - - - - - - - - distance calculation for within the cluster
    double calc_x1_internalDist(int indxCount, dataPoint<T>& x1_point, std::vector<dataPoint<T>>& dataSet)
    {
        // the sum of (point 1 - point 2) where point 2 is every point within the clust
        double x1_dist_within_clust = 0.0;         
            
        // - - - - get distance from current point and every other point in clust
        for(int i =0; i < indxCount; i++ )
        {
            dataPoint<T>& x2_point = dataSet.at(clust_data_indicies[i]);

            // --- don't compare x1 & x1
            if(x1_point.getDataID() != x2_point.getDataID())
            {
                // += (x1 - x2)^2
                x1_dist_within_clust += x1_point.calcSquaredEuclidDist(x2_point);
            }
        }

         return (x1_dist_within_clust / (indxCount - 1)); 
    }



    // - - - - - - - - Si distance calculation for the other clusters
    double calc_x1_externalDist(dataPoint<T>& x1_point, clust<T>& currClust, std::vector<dataPoint<T>>& dataSet)
    {
        std::vector<int> clustIndicies = currClust.getDataIndicies(); 
        int KlustIndxLen = clustIndicies.size(); 
        double x1_external_dist = 0.0; 

        // distance from x1 and every point for the other clusters
        for(int i =0; i < KlustIndxLen; i++)
        {
            dataPoint<T>& x2_point = dataSet.at(clustIndicies[i]); 
            x1_external_dist += x1_point.calcSquaredEuclidDist(x2_point);
        }

        return (x1_external_dist / KlustIndxLen); 
    }


    // = = = = = = = = = = = = = =  main calculations 
    // - - - - - - finds the mean data vector
    std::vector<T> genMeanDataVector( std::vector<dataPoint<T>>& dataSet)
    {
        // gen vector dimensions
        // true = include centroid data
        // false = exclude centroid data
        
        dataPoint<T>& temp = dataSet.at(0); 
        int numOfFeatures = temp.getDataVectorByRef().size(); 
    
        std::vector<double> meanFeatureVector(numOfFeatures, 0.0); 
    
        int sizeOfData = clust_data_indicies.size(); 

        // ------ go through each data vector
        for(int dataVector = 0; dataVector < sizeOfData; dataVector++)
        {
            int pos = clust_data_indicies.at(dataVector); 
            std::vector<T>& elms = dataSet.at(pos).getDataVectorByRef(); 
            
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


   void genClustSilhouetteScore(std::vector<dataPoint<T>>& dataSet, std::vector<clust<T>>& clustList, int numOfClust)
    {       
       
             
        int totalClustPoints = clust_data_indicies.size(); 
    
        // Handle single-point clusters
        if(totalClustPoints <= 1) 
        {
            total_SScore = 0.0;
            class_level_SScore = 0.0;
            return;
        }
    
        double totalClustSScores = 0.0; 

        // - - - - perform calculation for each point in the cluster
        for(int pos = 0; pos < totalClustPoints; pos++)
        {
            dataPoint<T>& x1_point = dataSet.at(clust_data_indicies[pos]);

            // for the current point x1
            // calculate the distance between x1 and every other point in *this* cluster 
            double x1_internal_ave = calc_x1_internalDist(totalClustPoints, x1_point, dataSet);

            // finds the distance between point x1 and every point in all k clusters
            std::vector<double> x1_externalDist; 
            x1_externalDist.reserve(numOfClust - 1);            
            
            int clust_distance = 0; 

            for(int k =0; k < numOfClust; k++)
            {
                if(clust_id != clustList.at(k).getID())
                {
                    x1_externalDist.push_back( calc_x1_externalDist(x1_point, clustList.at(k), dataSet));
                }
            }

            // finds the min average distance
            double min_x1_external_dist = x1_externalDist.at(0);  
            for(int k = 1; k < x1_externalDist.size(); k++)
            {
                if(min_x1_external_dist > x1_externalDist.at(k))
                {
                    min_x1_external_dist = x1_externalDist.at(k); 
                }
            }

            // calculate the silhouetteScore 
            double pt1 = min_x1_external_dist - x1_internal_ave;
            double pt2; 
            double x1_SScore; 
            
            // get max(Ai, Bi)
            if(x1_internal_ave > min_x1_external_dist) 
                { pt2 = x1_internal_ave;}

            else if(min_x1_external_dist > x1_internal_ave) 
                {pt2 = min_x1_external_dist; }

            else 
                {pt2 = x1_internal_ave; }

            x1_SScore = pt1 / pt2; 
            totalClustSScores += x1_SScore; 
        }
        total_SScore = totalClustSScores; 
        class_level_SScore = totalClustSScores / totalClustPoints;       
    }
    // - - - - - - - SSE calculation 
    void genSSE( std::vector<dataPoint<T>>& dataSet)
    {
        std::vector<T>& centroidFeatures = centroid.getDataVectorByRef(); 
        int sizeOfData = clust_data_indicies.size(); 

        double clustSSE = 0; 

        // ----- go through every dataPoint 
        for(int dataVec = 0; dataVec < sizeOfData; dataVec++ )
        {
            int currIndx = clust_data_indicies.at(dataVec); 
            std::vector<T>& elms = dataSet.at(currIndx).getDataVectorByRef(); 
            double dataVectorSSE = 0.0;

            // -------- find the total distance of a dataVector from centroid 
            for(int element = 0; element < elms.size(); element++  )
            {
                // (x1 - x2)^2 | x1 = dataPoint feature, x2 = centroid feature
                T x1 = elms.at(element);
                T x2 = centroidFeatures.at(element);
                
                // find distance from dataPoint feature and centroid feature
                double diff = x1 -x2; 
                diff = diff * diff;

                // calculate dataPoint's total distance from centroid
                dataVectorSSE += diff; 
            }
            clustSSE += dataVectorSSE; 
        }

        class_level_SSE = clustSSE; 
    }



    // = = = = = = = = = = = = = = constructors 
    clust()
    {
        class_level_SSE = double{};
        total_SScore = 0.0;
        class_level_SScore = 0.0; 
        centroid = dataPoint<T>{};
        clust_data_indicies = std::vector<int>{};
    }
};

#endif