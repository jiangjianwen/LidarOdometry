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

#include <cmath>
#include <vector>
#include <string>
#include "common.h"
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
#include <pcl/io/pcd_io.h>
#include "extractFeatures.hpp"
#include "laserOdometry.hpp"
#include "laserMapping.hpp"

std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    lidar_data_file.seekg(0, std::ios::end);
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    lidar_data_file.seekg(0, std::ios::beg);

    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char *>(&lidar_data_buffer[0]), num_elements * sizeof(float));
    return lidar_data_buffer;
}

class LidarOdometry
{
private:
    extractFeatures ef;
    laserOdometry lo;
    laserMapping lm;

public:
    Eigen::Quaterniond q;
    Eigen::Vector3d t;

    LidarOdometry() {}

    void process(pcl::PointCloud<pcl::PointXYZI> &laser_cloud)
    {
        ef.extractFeature(laser_cloud);

        lo.process(ef.cornerPointsSharp, ef.cornerPointsLessSharp, ef.surfPointsFlat, ef.surfPointsLessFlat);

        lm.process(ef.cornerPointsLessSharp, ef.surfPointsLessFlat, lo.q_odom, lo.t_odom);

        q = lm.q_w_curr;
        t = lm.t_w_curr;
    }
};

int main(int argc, char **argv)
{

    std::ifstream timestamp_file("/home/jianwen/data/KITTI/odometry/00/times.txt", std::ifstream::in);

    std::string line;
    std::size_t line_num = 0;

    LidarOdometry LO;

    while (std::getline(timestamp_file, line))
    {
        std::stringstream lidar_data_path;
        lidar_data_path << "/home/jianwen/data/KITTI/odometry/00/semantic/" << line_num << ".bin";
        std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());
        std::cout << "totally " << lidar_data.size() / 4.0 << " points in this lidar frame " << line_num << std::endl;

        pcl::PointCloud<pcl::PointXYZI> laser_cloud;
        for (std::size_t i = 0; i < lidar_data.size(); i += 4)
        {
            pcl::PointXYZI point;
            point.x = lidar_data[i];
            point.y = lidar_data[i + 1];
            point.z = lidar_data[i + 2];
            point.intensity = lidar_data[i + 3];
            laser_cloud.push_back(point);
        }

        laser_cloud.width = 1;
        laser_cloud.height = laser_cloud.points.size();

        LO.process(laser_cloud);

        line_num++;
    }

    return 0;
}
