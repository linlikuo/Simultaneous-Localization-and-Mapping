opencv-python>=3.6 matplotlib>=2.2 
numpy>=1.14 imageio
1. load_data.py is used for data loading including Encoders, Imu, Lidar and the RGBD.
2. utils.py contains softmax, stratified resample, time align, mapping, and texture mapping.
   time align is the function that we used to align two time stamp.
3. map_utils.py has the function mapcorrelation and bresenham2D.
   bresenham2D is the function that we used to find all points between start point and end point
4. Execute the program with python main.py 
   and you can change dataset by changing the parameter value in main.py