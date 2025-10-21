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
    double distance_from_cluster; 
    std::vector<T> data_point;
   
   
    // = = = = = = = = = = = = = = setters and getters 
    void setID(std::string id) { dataID = id; }
    void updateDistanceFromClust(double distance) { distance_from_cluster = distance; }

    std::string getDataID(){ return dataID;}
    double getDistanceFromClust(){return distance_from_cluster;}
    std::vector<T> getDataVector(){ return data_point;}

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
    dataPoint(std::vector<T> dataPoint, double distanceFromClust, std::string id)
    {
        dataID = id; 
        data_point = dataPoint;
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

    clusterData is the data from the DataSet
    that this specific cluster is holding

    the Template T is to be passed down to the 
    DataPoint class, so that it knows what the
    DataVectors are storing. 
*/
template <class T>
class clust 
{

    double class_level_SSE; 
    dataPoint<T> centroid; 
    std::vector<dataPoint<T>> clusterData; 

    public:
    // = = = = = = = = = = = = = =  setters and getters    

    void addDataPoint(std::vector<T>& data, std::string pointID)
    { 
        clusterData.push_back(dataPoint<T>(data, 0.0, pointID));
    }
    void addDataPoint(dataPoint<T>& data)
    {
        clusterData.push_back(data);
    }

    void setCentroid(dataPoint<T>& cent) { centroid = cent;}

    dataPoint<T> getCentroid(){return centroid; }
    double getClassLevelSSE(){return class_level_SSE;}
    std::vector<dataPoint<T>> getClusterData(){return clusterData; }

    // = = = = = = = = = = = = = = visualization methods

    void showCentroid()
    {
        centroid.printData(); 
        std::cout << std::endl; 
    }
    void logCentroids(std::ofstream& file)
    {
            centroid.logData(file); 
            file << std::endl; 
    }
    void displayDataAt(int index)
    {
        clusterData.at(index).printData();   
    }

    // = = = = = = = = = = = = = = some methods <IDK what to label these>
    void resetCluster()
    {
        centroid = dataPoint<T>{};
        class_level_SSE = 0.0; 
        clusterData = std::vector<dataPoint<T>>{};

    }
    std::vector<T> centroidUsingMean()
    {
     
        int numOfFeatures = centroid.getDataVector().size(); 
        // initialize a vector of the proper dimensions with default value 0.0
        std::vector<double> new_centroid(numOfFeatures, 0.0); 
        int sizeOfData = clusterData.size(); 

        // ------ go through each data vector
        for(int dataVector = 0; dataVector < sizeOfData; dataVector++)
        {
            std::vector<T> elms = clusterData.at(dataVector).getDataVector(); 
            
            // -------- calculate the sum of each feature
           for(int element = 0; element < elms.size(); element++  )
            {
                new_centroid.at(element) += elms.at(element); 
            }

        }

        // ------- divide each feature's sum by total dataPoints
        // ------- this creates a vector of means
        for(int i =0; i < new_centroid.size(); i++)
        {
            new_centroid.at(i) = new_centroid.at(i) / sizeOfData; 
        }

        // ---- return new centroid's dataVector
        return new_centroid; 
    }
    void genSSE()
    {
        std::vector<T> centroidFeatures = centroid.getDataVector(); 
        int sizeOfData = clusterData.size(); 

        double clustSSE = 0; 

        // ----- go through every dataPoint 
        for(int dataVec = 0; dataVec < sizeOfData; dataVec++ )
        {
            std::vector<T> elms = clusterData.at(dataVec).getDataVector(); 
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
        centroid = dataPoint<T>{};
        clusterData = std::vector<dataPoint<T>>{};
    }

};

//!!!!!!!!!!!!!!!!!!!!!! TODO
/*
[] add SSE calculation to cluster class
[] move visualization & log methods from cluster class to controller
[] look over controller class to make sure it's all good
[] write the controller class methods 
[] connect controller class to main 
*/


#endif