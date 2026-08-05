#Some sections of code were taken from Chris Baird - Ontario Tech MARS Lab 

import numpy as np
import cv2
import scipy
import time

UNKNOWN = 128
#Evaluate Known Map
#Determine Frontier Points
#Evaluate Frontier Points - some time and hazard cost
#Select Frontier Point

# -- Reveal Handelers --
def reveal_grid(robot_map, map, pose, search_area):
    robot_map_prev = robot_map.copy()
    reveal_area = np.zeros(map.shape, np.uint8)
    reveal_area = cv2.circle(reveal_area, (pose[1], pose[0]), search_area, 255, -1)
    contours, _ = cv2.findContours(reveal_area, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    reveal_area = cv2.bitwise_and(map, reveal_area)
    for circ_point in contours[0]:
        
        #obscured = False
        line = bresenham(pose[0], pose[1], circ_point[0,1], circ_point[0,0])
        for l_point in line:
            if robot_map_prev[l_point] == UNKNOWN:
                robot_map[l_point] = reveal_area[l_point]
            if reveal_area[l_point] == 0:
                break
                        
    return robot_map

def bresenham(x1, y1, x2, y2):
    points = []
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
            
    return points





# -- Main --
def main():

    # -- Map Construction --
    Maps = [r"Maps/hall.png", r"Maps/house.png", r"Maps/house2.png"]
    Map = cv2.imread(Maps[0])
    Map_display = Map.copy() #Map Copy for display purposes
    Map = cv2.cvtColor(Map, cv2.COLOR_RGB2GRAY) #Map in grayscale for processing

    search_radius = 15

    # pad image with zeros to avoid index out of bounds error when searching for frontier points
    full_image = np.zeros(
        (Map.shape[0] + 2 * search_radius, Map.shape[1] + 2 * search_radius),
        np.uint8,
    )
    full_image[
        search_radius : search_radius + Map.shape[0],
        search_radius : search_radius + Map.shape[1],
    ] = Map

    image = np.zeros(
        (Map.shape[0] + 2 * search_radius, Map.shape[1] + 2 * search_radius),
        np.uint8,
    )
    image[
        search_radius : search_radius + Map.shape[0],
        search_radius : search_radius + Map.shape[1],
    ] = 127 * np.ones(Map.shape, np.uint8)

    # -- Robot Construction --
    robot_map = np.ones(np.shape(Map), dtype=np.uint8) * UNKNOWN
    start_pos = [249, 125]
    print(start_pos)
    robot_pos = start_pos

    robot_map = reveal_grid(robot_map, Map, robot_pos, search_radius)
    while True:

        # evaluate frontiers and select one
        #go to frontier

        
        robot_map = reveal_grid(robot_map, Map, robot_pos, search_radius)
        cv2.imshow("Map", robot_map)
        cv2.waitKey(1)

    
          

if __name__ == "__main__":
    main()