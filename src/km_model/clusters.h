#ifndef CLUST_STUFF 
#define  CLUST_STUFF

#include <iostream>
#include <string>
#include <vector>
#include "core/algorithm_backend.h"

//~ ================================================= dataPoints structure
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
// TODO - - - - - change name to something clearer like: data_element

{
    // data ID | distance from nearest cluster | data
    std::string dataPoint_class_id; 
    std::vector<T> feature_vector;
    double distance_from_cluster; 
    algorithm_backend<T> metrics; 
   
   
    // = = = = = = = = = = = = = = setters and getters 

    // - - - - - - - - - - data point ID 
    void setPointID(std::string& id) 
        { dataPoint_class_id = std::move(id); }
    
    std::string& getPointID()
        { return dataPoint_class_id;}

    // - - - - - - - - - distance from cluster
    void setDistanceFromCluster(double dist_from_clust)
        { distance_from_cluster = dist_from_clust}

    double getDistanceFromCluster()
        {return distance_from_cluster}

    // ! ------------------------------------------ names had Data instead of Feature | setFeatureVector was setDataVector
    // - - - - - - - - - - - data vector 
    void setFeatureVector(std::vector<T>& updatedVector)
        { feature_vector = std::move(updatedVector); }
    
    std::vector<T> getFeatureVector_copy()
        { return feature_vector;}

    
    std::vector<T>& getFeatureVector_ref()
        { return feature_vector; }    

    int getFeatureDimensions()
        { return feature_vector.size(); }
    // ! ------------------------------------------

    // = = = = = = = = = = = = = = Metrics  

    double find_euclid_dist(dataPoint<T>& x2_dataPoint)
    {
        std::vector<T>& x2_feature_vector = x2_dataPoint.getFeatureVector_ref(); 
        
        return metrics.sqr_euclid_dist(feature_vector, x2_feature_vector);
    }   

    double find_euclid_dist(std::vector<T>& x2_feature_vector)
    {
        return metrics.sqr_euclid_dist(feature_vector, x2_feature_vector); 
    }

   
    // = = = = = = = = = = = = = = constructors 
    dataPoint(std::vector<T>& feature_vector, std::string& instance_id)
    :   dataPoint_class_id(std::move(instance_id)),
        feature_vector(std::move(feature_vector)),
        metrics(),  
        distance_from_cluster()
    {}
    
    dataPoint()
    :   dataPoint_class_id(),
        feature_vector(),
        metrics(), 
        distance_from_cluster()
    {}
};


//~ ========================================================== cluster class 
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
    algorithm_backend<T> metrics;
    std::string clust_id; 
    dataPoint<T> centroid; 
    std::vector<int> clust_data_indicies;


    public: 
    //  = = = = = = = = = = = = = = = = = = = = =  setters and getters    

    // - - - - - - - - - - - - indicies of datapoints assigned to cluster
    void assignData(int dataSetIndex)
        { clust_data_indicies.push_back(dataSetIndex);}


    std::vector<int> getAssignedData_copy()
        {return clust_data_indicies; }

    std::vector<int>& getAssignedData_ref()
        {return clust_data_indicies; }


    int getSize()
        {return clust_data_indicies.size();}


    // - - - - - - - - - - -  centroid 
    void setCentroid(dataPoint<T>& cent) 
        { centroid = cent;}
    
    dataPoint<T>& getCentroid_ref()
        {return centroid; }

    dataPoint<T> getCentroid_copy()
        {return centroid}


    // - - - - - - - - - - - cluster ID
    void setID(int id)
        { clust_id = "c" + std::to_string(id); }

    std::string& getID_ref()
        {return clust_id;}
    
    std::string getID_copy()
        {return clust_id;}


    // - - - - - - - - - - - class SSE
    double getClassLevelSSE()
        {return class_level_SSE;}


    // - - - - - - - - - - - class shiloette score 
    double getSScoreContribution()
        {return total_SScore;}
    
    // - - - - - - - - - - - reset clust to default values
    void resetCluster()
    {
        centroid = dataPoint<T>{};
        class_level_SSE = 0.0; 
        total_SScore = 0.0; 
        class_level_SScore = 0.0; 
        clust_data_indicies.clear();
    }


    // - - - - - - - - - - - - - -  - SSE calculation 
    void sumSquaredError( std::vector<dataPoint<T>>& dataSet)
    {
        // - - initialize variables
        std::vector<T>& centroidFeatures = centroid.getFeatureVector_ref(); 
        int totalPointsInCluster = clust_data_indicies.size(); 

        double clustSSE = 0; 

        // - - goes through each data point assigned to cluster 
        for(int currPoint = 0; currPoint < totalPointsInCluster; currPoint++ )
        {
            int pos = clust_data_indicies.at(currPoint); 
            std::vector<T>& currFeatures = dataSet.at(pos).getFeatureVector_ref(); 

            double currFeatureVectorSSE = metrics.sqr_euclid_dist(currFeatures , centroidFeatures);

            clustSSE += currFeatureVectorSSE; 
        }

        class_level_SSE = clustSSE; 
    }


    // - - - - - - finds the mean data vector
    std::vector<T> genMeanFeatVector( std::vector<dataPoint<T>>& dataSet)
    {
        
        // - - initialize variables | finalMeanVector - vector of each column's mean
        int featureVectorDimensions = dataSet.at(0).getFeatureDimensions(); 
    
        std::vector<double> finalMeanVector(featureVectorDimensions, 0.0); 
    
        // ~ cluster_Data_indicies - vector of indicies for each point assigned to the cluster
        int totalPointsInCluster = clust_data_indicies.size(); 

        // - - goes through data assigned to cluster and sums each column
        for(int currentPoint = 0; currentPoint < totalPointsInCluster; currentPoint++)
        {
            int pos = clust_data_indicies.at(currentPoint); 
    
            std::vector<T>& features_currentPoint = dataSet.at(pos).getFeatureVector_ref(); 
            
            // sums columns 
           for(int currFeature = 0; currFeature < features_currentPoint.size(); currFeature++)
            {
                finalMeanVector.at(currFeature) += features_currentPoint.at(currFeature); 
            }
        }

        // divide each column's sum by the total amount of points
        for(int i =0; i < meanFeatureVector.size(); i++)
        {
            meanFeatureVector.at(i) = meanFeatureVector.at(i) / totalPointsInCluster; 
        }

        // return vector of column means
        return finalMeanVector; 
    }

    // - - - - - - cohesion using a comprehensive point to point strategy
    //&             a(i) = (currPoint - every_Other_Point_In_Clust)^2
    double clust_cohesion_pairwise(std::vector<dataPoint<T>>& dataset)
    {
        int currClustTotalPoints = clust_data_indicies.size(); 
        double currPointCohesionCoeff = 0.0; 

        // --- loop through curr clust assigned points
        for(int currPoint = 0; currPoint < currClustTotalPoints; currPoint++)
        {
            int x1_pos = clust_data_indicies.at(currPoint); 

            // --- (x_1 - x_2)^2 | x_2 = each point assigned to clust excluding x_1
            for(int allPoints = currPoint + 1; allPoints < currClustTotalPoints; allPoints++)
            {
                int x2_pos = clust_data_indicies.at(allPoints); 
            
                
                std::vector<T>& x1_features = dataset.at(x1_pos).getFeatureVector_ref; 
                std::vector<T>& x2_features = dataset.at(x2_pos).getFeatureVector_ref; 

                currPointCohesionCoeff += metrics.sqr_euclid_dist(x1_features, x2_features); 
                    
                
            }

        }
        double divisor = currClustTotalPoints * ((currClustTotalPoints -1) / 2.0);
        
        return (currPointCohesionCoeff / divisor);
    }

    // - - - - - - separation using point to point, pairwise, strategy
    //&            s(i) = (currPoint - every_Other_Point_In_other_clusters)^2
    double separation_pairwise(std::vector<dataPoint<T>>& dataset, std::vector<int>& clustB_data)
    {
        int clustA_TotalPoints = clust_data_indicies.size(); 
        int clustB_TotalPoints = clustB_data.size(); 

        double totalSeparation = 0.0; 

        // --- loop through clustA assigned points
        for(int currPoint = 0; currPoint < clustA_TotalPoints; currPoint++)
        {
            int x1_pos = clust_data_indicies.at(currPoint); 

            // --- (x_1 - x_2)^2 | loop through clustB
            for(int allPoints = 0; allPoints < clustB_TotalPoints; allPoints++)
            {
                int x2_pos = clustB_data.at(allPoints); 
            
                
                std::vector<T>& x1_features = dataset.at(x1_pos).getFeatureVector_ref; 
                std::vector<T>& x2_features = dataset.at(x2_pos).getFeatureVector_ref; 

                totalSeparation += metrics.sqr_euclid_dist(x1_features, x2_features); 
                    
                
            }

        }
        double divisor = clustA_TotalPoints * clustB_TotalPoints;
        
        return (totalSeparation / divisor);

    }

    // = = = = = = = = = = = = = = constructors 
    clust()
    : metrics()
    {
        class_level_SSE = double{};
        total_SScore = 0.0;
        class_level_SScore = 0.0; 
        centroid = dataPoint<T>{};
        clust_data_indicies = std::vector<int>{};
    }
};

#endif