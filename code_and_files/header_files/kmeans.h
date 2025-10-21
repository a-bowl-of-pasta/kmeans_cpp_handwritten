#ifndef KMEANS
#define KMEANS

#include <iostream>
#include <string>
#include <vector>
   
// ================================================= dataPoints structure
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
template <class T>
class k_means 
{

    double class_level_SSE; 
    dataPoint centroid; 
    std::vector<dataPoint> clusterData; 

    public:
    // = = = = = = = = = = = = = =  setters and getters    

    void add_dataPoint(std::vector<T>& data, std::string pointID)
        { clusterData.push_back(dataPoint(data, 0.0, pointID));}
    
    void set_Centroid(dataPoint& cent) { centroid = cent;}

    dataPoint getCentroid(){return centroid; }
    double getClassLevelSSE(){return class_level_SSE;}
    std::vector<dataPoint> getClusterData(){return clusterData; }

    // = = = = = = = = = = = = = = visualization methods

    void show_centroid()
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

    // = = = = = = = = = = = = = = constructors 
    k_means()
    {
        class_level_SSE = double{};
        centroid = dataPoint{};
        clusterData = std::vector<dataPoint>{};
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