#Some sections of code were taken from Chris Baird - Ontario Tech MARS Lab 

import numpy as np
import cv2
import scipy


#Evaluate Known Map
#Determine Frontier Points
#Evaluate Frontier Points - some time and hazard cost
#Select Frontier Point

def reveal_grid(robot_map, map, pose, search_area):
    robot_map_prev = robot_map.copy()
    reveal_area = np.zeros(map.shape, np.uint8)
    reveal_area = cv2.circle(reveal_area, (pose[0], pose[1]), search_area, 255, -1)
    contours = cv2.findContours()

    reveal_area = cv2.bitwise_and(map, reveal_area)

    #Get circle circumference points from find contours
    #grab each point on circle circumference and draw bresenham line
    #Loop through bresenham points
    #if point is black(occupied) add it to robot_map and set obscured flag to true
    #if obscured flag is false and point is white(unoccupied) add it to robot_map

def bresenham(x1, y1, x2, y2):
    points = []
    m_new = 2 * (y2 - y1)
    slope_error_new = m_new - (x2 - x1)

    y = y1
    for x in range(x1, x2+1):

        points.append((x,y))

        # Add slope to increment angle formed
        slope_error_new = slope_error_new + m_new

        # Slope error reached limit, time to
        # increment y and update slope error.
        if (slope_error_new >= 0):
            y = y+1
            slope_error_new = slope_error_new - 2 * (x2 - x1)
    return points

def main():

    # -- Map Construction --
    Maps = [r"Maps/hall.png", r"Maps/house.png", r"Maps/house2.png"]
    Map = cv2.imread(Maps[0])
    Map_display = Map.copy() #Map Copy for display purposes
    Map = cv2.cvtColor(Map, cv2.COLOR_RGB2GRAY) #Map in grayscale for processing

    search_radius = 50

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
    robot_map = np.ones(np.shape(Map), dtype=np.uint8) * 128
    start_pos = [0,0]
    robot_pos = start_pos

    while True:

        cv2.imshow("Map", Map_display)
        cv2.waitKey(100)


if __name__ == "__main__":
    main()