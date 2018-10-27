import os
import numpy as np
 
def parse_nvm_3d_points(nvm_file):
    """
    Formats of nvm file:
        <Number of cameras>   <List of cameras>
        <Number of 3D points> <List of points>
        <Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>
        <Measurement> = <Image index> <Feature Index> <xy>
    """
    cams = []       # List image frames 
    cam_points = {} # Map key: index of frame, value: list of indices of 3d points that are visible to this frame.
    points = []     # List of 3d points in the reconstruction model
    
    with open(nvm_file, 'r') as f:
        f.readline()    # Skip headding lines
        f.readline()
        cam_num = int(f.readline().split()[0])
        for i in range(cam_num):
            line = f.readline()
            frame = line.split()[0]
            cams.append(frame)
            cam_points[frame] = []
        f.readline()
        point_num = int(f.readline().split()[0])
        for i in range(point_num):
            line = f.readline()
            cur = line.split()
            X = [float(cur[0]),float(cur[1]),float(cur[2])]
            points.append(X)
            measure_num = int(cur[6])
            for j in range(measure_num):
                idx = int(cur[7+j*4])
                frame = cams[idx]
                cam_points[frame].append(i)
    points = np.asarray(points)
    return (points, cam_points)
