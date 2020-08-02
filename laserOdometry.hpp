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
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <eigen3/Eigen/Dense>
#include <queue>
#include "common.h"
#include "lidarFactor.hpp"

class laserOdometry
{
private:
    int corner_correspondence, plane_correspondence;

    bool systemInited;

    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;

    int laserCloudCornerLastNum, laserCloudSurfLastNum;

    Eigen::Quaterniond q_w_curr;
    Eigen::Vector3d t_w_curr;

    double para_q[4] = {0, 0, 0, 1};
    double para_t[3] = {0, 0, 0};

    Eigen::Map<Eigen::Quaterniond> q_last_curr = Eigen::Map<Eigen::Quaterniond>(para_q);
    Eigen::Map<Eigen::Vector3d> t_last_curr = Eigen::Map<Eigen::Vector3d>(para_t);

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp, cornerPointsLessSharp, surfPointsFlat, surfPointsLessFlat;

    void TransformToStart(PointType const *const pi, PointType *const po)
    {
        double s = 1.0;
        Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
        Eigen::Vector3d t_point_last = s * t_last_curr;
        Eigen::Vector3d point(pi->x, pi->y, pi->z);
        Eigen::Vector3d un_point = q_point_last * point + t_point_last;

        po->x = un_point.x();
        po->y = un_point.y();
        po->z = un_point.z();
        po->intensity = pi->intensity;
    }

public:
    Eigen::Quaterniond q_odom;
    Eigen::Vector3d t_odom;

    laserOdometry()
    {
        corner_correspondence = 0;
        plane_correspondence = 0;

        systemInited = false;

        kdtreeCornerLast.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        kdtreeSurfLast.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());

        laserCloudCornerLastNum = 0;
        laserCloudSurfLastNum = 0;

        q_w_curr = Eigen::Quaterniond(1, 0, 0, 0);
        t_w_curr = Eigen::Vector3d(0, 0, 0);

        cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
        cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
        surfPointsFlat.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());
    }
    ~laserOdometry() {}

    void process(pcl::PointCloud<PointType>::Ptr cornerPointsSharpIn, pcl::PointCloud<PointType>::Ptr cornerPointsLessSharpIn,
                 pcl::PointCloud<PointType>::Ptr surfPointsFlatIn, pcl::PointCloud<PointType>::Ptr surfPointsLessFlatIn)
    {
        cornerPointsSharp->points.clear();
        cornerPointsLessSharp->points.clear();
        surfPointsFlat->points.clear();
        surfPointsLessFlat->points.clear();

        *cornerPointsSharp = *cornerPointsSharpIn;
        *cornerPointsLessSharp = *cornerPointsLessSharpIn;
        *surfPointsFlat = *surfPointsFlatIn;
        *surfPointsLessFlat = *surfPointsLessFlatIn;

        if (!systemInited)
        {
            systemInited = true;
            std::cout << "Initialization finished \n";
        }
        else
        {
            int cornerPointsSharpNum = cornerPointsSharp->points.size();
            int surfPointsFlatNum = surfPointsFlat->points.size();

            pcl::PointCloud<pcl::PointNormal> data_pi;
            pcl::PointCloud<pcl::PointNormal> model_qi;

            for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
            {
                corner_correspondence = 0;
                plane_correspondence = 0;

                //ceres::LossFunction *loss_function = NULL;
                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::LocalParameterization *q_parameterization =
                    new ceres::EigenQuaternionParameterization();
                ceres::Problem::Options problem_options;

                ceres::Problem problem(problem_options);
                problem.AddParameterBlock(para_q, 4, q_parameterization);
                problem.AddParameterBlock(para_t, 3);

                pcl::PointXYZI pointSel;
                std::vector<int> pointSearchInd;
                std::vector<float> pointSearchSqDis;

                // find correspondence for corner features
                for (int i = 0; i < cornerPointsSharpNum; ++i)
                {
                    TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
                    kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                    int closestPointInd = -1, minPointInd2 = -1;
                    if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                    {
                        closestPointInd = pointSearchInd[0];
                        int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                        double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                        // search in the direction of increasing scan line
                        for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                        {
                            // if in the same scan line, continue
                            if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                continue;

                            // if not in nearby scans, end the loop
                            if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                    (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                    (laserCloudCornerLast->points[j].z - pointSel.z);

                            if (pointSqDis < minPointSqDis2)
                            {
                                // find nearer point
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }

                        // search in the direction of decreasing scan line
                        for (int j = closestPointInd - 1; j >= 0; --j)
                        {
                            // if in the same scan line, continue
                            if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                continue;

                            // if not in nearby scans, end the loop
                            if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                    (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                    (laserCloudCornerLast->points[j].z - pointSel.z);

                            if (pointSqDis < minPointSqDis2)
                            {
                                // find nearer point
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }
                    }
                    if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                    {
                        Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                   cornerPointsSharp->points[i].y,
                                                   cornerPointsSharp->points[i].z);
                        Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                     laserCloudCornerLast->points[closestPointInd].y,
                                                     laserCloudCornerLast->points[closestPointInd].z);
                        Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                     laserCloudCornerLast->points[minPointInd2].y,
                                                     laserCloudCornerLast->points[minPointInd2].z);

                        double s;

                        s = 1.0;
                        ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                        corner_correspondence++;
                    }
                }

                //                    pcl::PointCloud<pcl::PointNormal> data_pi;
                data_pi.points.clear();
                //                    pcl::PointCloud<pcl::PointNormal> model_qi;
                model_qi.points.clear();

                // find correspondence for plane features
                for (int i = 0; i < surfPointsFlatNum; ++i)
                {
                    TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                    kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                    int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                    if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                    {
                        closestPointInd = pointSearchInd[0];

                        // get closest point's scan ID
                        int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                        double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                        // search in the direction of increasing scan line
                        for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                        {
                            // if not in nearby scans, end the loop
                            if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                    (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                    (laserCloudSurfLast->points[j].z - pointSel.z);

                            // if in the same or lower scan line
                            if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                            {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                            // if in the higher scan line
                            else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                            {
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }

                        // search in the direction of decreasing scan line
                        for (int j = closestPointInd - 1; j >= 0; --j)
                        {
                            // if not in nearby scans, end the loop
                            if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                    (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                    (laserCloudSurfLast->points[j].z - pointSel.z);

                            // if in the same or higher scan line
                            if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                            {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                            else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                            {
                                // find nearer point
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }

                        if (minPointInd2 >= 0 && minPointInd3 >= 0)
                        {

                            Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                       surfPointsFlat->points[i].y,
                                                       surfPointsFlat->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                         laserCloudSurfLast->points[closestPointInd].y,
                                                         laserCloudSurfLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                         laserCloudSurfLast->points[minPointInd2].y,
                                                         laserCloudSurfLast->points[minPointInd2].z);
                            Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                         laserCloudSurfLast->points[minPointInd3].y,
                                                         laserCloudSurfLast->points[minPointInd3].z);

                            double s;

                            s = 1.0;
                            ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            plane_correspondence++;

                            pcl::PointNormal d_p;
                            d_p.x = curr_point[0];
                            d_p.y = curr_point[1];
                            d_p.z = curr_point[2];
                            data_pi.points.push_back(d_p);

                            pcl::PointNormal m_qi;
                            m_qi.x = last_point_a[0];
                            m_qi.y = last_point_a[1];
                            m_qi.z = last_point_a[2];

                            Eigen::Vector3d ljm_norm;
                            ljm_norm = (last_point_a - last_point_b).cross(last_point_a - last_point_c);
                            ljm_norm.normalize();
                            m_qi.normal_x = ljm_norm.x();
                            m_qi.normal_y = ljm_norm.y();
                            m_qi.normal_z = ljm_norm.z();

                            model_qi.points.push_back(m_qi);
                        }
                    }
                }

                if ((corner_correspondence + plane_correspondence) < 10)
                {
                    printf("less correspondence! *************************************************\n");
                }

                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 4;
                options.minimizer_progress_to_stdout = false;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
            }

            t_w_curr = t_w_curr + q_w_curr * t_last_curr;
            q_w_curr = q_w_curr * q_last_curr;
        }

        q_odom = q_w_curr;
        t_odom = t_w_curr;

        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        laserCloudSurfLastNum = laserCloudSurfLast->points.size();

        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
    }
};