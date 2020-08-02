// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <opencv/cv.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Dense>
#include "common.h"

using std::atan2;
using std::cos;
using std::sin;

float cloudCurvature[400000];
bool comp(int i, int j) { return (cloudCurvature[i] < cloudCurvature[j]); }

class extractFeatures
{
private:
    int cloudSortInd[400000];
    int cloudNeighborPicked[400000];
    int cloudLabel[400000];

    template <typename PointT>
    void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                                pcl::PointCloud<PointT> &cloud_out, float thres)
    {
        if (&cloud_in != &cloud_out)
        {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize(cloud_in.points.size());
        }

        size_t j = 0;

        for (size_t i = 0; i < cloud_in.points.size(); ++i)
        {
            if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
                continue;
            cloud_out.points[j] = cloud_in.points[i];
            j++;
        }
        if (j != cloud_in.points.size())
        {
            cloud_out.points.resize(j);
        }

        cloud_out.height = 1;
        cloud_out.width = static_cast<uint32_t>(j);
        cloud_out.is_dense = true;
    }

    void extractStablelaserCloud(pcl::PointCloud<pcl::PointXYZI> &laserCloudIn, pcl::PointCloud<pcl::PointXYZI> &stablelaserCloud, pcl::PointCloud<pcl::PointXYZI> &notstablelaserCloud)
    {
        stablelaserCloud.points.clear();
        notstablelaserCloud.points.clear();

        for (std::size_t i = 0; i < laserCloudIn.points.size(); i++)
        {
            int label = int(laserCloudIn.points[i].intensity);
            // 70 vegetable 72 terrain 20 "other-vehicle" 30: "person" 31: "bicyclist" 32: "motorcyclist" 11: "bicycle" 15: "motorcycle"
            if (label == 72 || label == 70 || label == 20 || label == 30 || label == 31 || label == 32 || label == 11 || label == 15)
                notstablelaserCloud.push_back(laserCloudIn.points[i]);

            else
            {
                stablelaserCloud.push_back(laserCloudIn.points[i]);
            }
        }
    }

public:
    pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
    pcl::PointCloud<PointType>::Ptr surfPointsFlat;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;
    pcl::PointCloud<PointType>::Ptr laserPointsFull;

    extractFeatures(/* args */)
    {
        cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
        cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
        surfPointsFlat.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());
        laserPointsFull.reset(new pcl::PointCloud<PointType>());
    }
    ~extractFeatures() {}

    void extractFeature(pcl::PointCloud<pcl::PointXYZI> &laserCloudOri)
    {
        cornerPointsSharp->points.clear();
        cornerPointsLessSharp->points.clear();
        surfPointsFlat->points.clear();
        surfPointsLessFlat->points.clear();
        laserPointsFull->points.clear();

        std::vector<int> scanStartInd(N_SCANS, 0);
        std::vector<int> scanEndInd(N_SCANS, 0);

        pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
        laserCloudIn.width = 1;
        laserCloudIn.height = laserCloudOri.points.size();

        laserCloudIn = laserCloudOri;

        std::vector<int> indices;

        pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
        removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);

        laserPointsFull = laserCloudIn.makeShared();

        int cloudSize = laserCloudIn.points.size();
        float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
        float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                              laserCloudIn.points[cloudSize - 1].x) +
                       2 * M_PI;

        if (endOri - startOri > 3 * M_PI)
        {
            endOri -= 2 * M_PI;
        }
        else if (endOri - startOri < M_PI)
        {
            endOri += 2 * M_PI;
        }

        pcl::PointCloud<pcl::PointXYZI> stablelaserCloud;
        pcl::PointCloud<pcl::PointXYZI> notstablelaserCloud;
        extractStablelaserCloud(laserCloudIn, stablelaserCloud, notstablelaserCloud);

        bool halfPassed = false;
        cloudSize = stablelaserCloud.points.size();
        int count = cloudSize;
        PointType point;
        std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
        for (int i = 0; i < cloudSize; i++)
        {
            point.x = stablelaserCloud.points[i].x;
            point.y = stablelaserCloud.points[i].y;
            point.z = stablelaserCloud.points[i].z;
            // point.intensity = stablelaserCloud.points[i].intensity;

            float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
            int scanID = 0;

            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }

            float ori = -atan2(point.y, point.x);
            if (!halfPassed)
            {
                if (ori < startOri - M_PI / 2)
                {
                    ori += 2 * M_PI;
                }
                else if (ori > startOri + M_PI * 3 / 2)
                {
                    ori -= 2 * M_PI;
                }

                if (ori - startOri > M_PI)
                {
                    halfPassed = true;
                }
            }
            else
            {
                ori += 2 * M_PI;
                if (ori < endOri - M_PI * 3 / 2)
                {
                    ori += 2 * M_PI;
                }
                else if (ori > endOri + M_PI / 2)
                {
                    ori -= 2 * M_PI;
                }
            }

            float relTime = (ori - startOri) / (endOri - startOri);
            point.intensity = scanID + scanPeriod * relTime;
            laserCloudScans[scanID].push_back(point);
        }

        cloudSize = count;
        //    printf("points size %d \n", cloudSize);

        pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < N_SCANS; i++)
        {
            scanStartInd[i] = laserCloud->size() + 5;
            *laserCloud += laserCloudScans[i];
            scanEndInd[i] = laserCloud->size() - 6;
        }

        //    printf("prepare time %f \n", t_prepare.toc());

        for (int i = 5; i < cloudSize - 5; i++)
        {
            float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
            float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
            float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

            cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
            cloudSortInd[i] = i;
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
        }

        for (int i = 0; i < N_SCANS; i++)
        {
            if (scanEndInd[i] - scanStartInd[i] < 6)
                continue;
            pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
            for (int j = 0; j < 6; j++)
            {
                int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
                int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

                std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, comp);

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSortInd[k];

                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] > 0.1)
                    {

                        largestPickedNum++;
                        if (largestPickedNum <= 2)
                        {
                            cloudLabel[ind] = 2;
                            cornerPointsSharp->push_back(laserCloud->points[ind]);
                            cornerPointsLessSharp->push_back(laserCloud->points[ind]);
                        }
                        else if (largestPickedNum <= 20)
                        {
                            cloudLabel[ind] = 1;
                            cornerPointsLessSharp->push_back(laserCloud->points[ind]);
                        }
                        else
                        {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSortInd[k];

                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] < 0.1)
                    {

                        cloudLabel[ind] = -1;
                        surfPointsFlat->push_back(laserCloud->points[ind]);

                        smallestPickedNum++;
                        if (smallestPickedNum >= 4)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0)
                    {
                        surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                    }
                }
            }

            pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
            pcl::VoxelGrid<PointType> downSizeFilter;
            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
            downSizeFilter.filter(surfPointsLessFlatScanDS);

            *surfPointsLessFlat += surfPointsLessFlatScanDS;
        }
    }
};
