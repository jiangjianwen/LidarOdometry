# LidarOdometry
## LidarOdometry is modified from LOAM

The changes are as follows:
1. remove ros.
2. "scan-to-map" is executed immediately after "scan-to-scan".

**Modifier:** [Jianwen Jiang]


## 1. Prerequisites
### 1.1 **Ubuntu**
Ubuntu 64-bit 16.04 or 18.04.

### 1.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).

### 1.3. **PCL**
Follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html).


## 2. Build LidarOdometry

```
    git clone https://github.com/jiangjianwen/LidarOdometry.git
    cd ./LidarOdometry
    make build
    cd build
    cmake ..
    make -j2
```

## 3.Acknowledgements
Thanks for LOAM(J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time), [LOAM_NOTED](https://github.com/cuitaixiang/LOAM_NOTED) and [A-LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM).
