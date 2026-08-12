#Some sections of code were taken from Chris Baird - Ontario Tech MARS Lab 

import numpy as np
import cv2
import scipy
import time
import math
import a_star
import copy

UNKNOWN = 127
HAZARD = 191

# -- Reveal Handelers --
#Reveals the map to the robot position with line of sight based projections 
def reveal_grid(robot_map, map, pose, search_area):
    #Draw a circle around the robot position and project lines to ever pixel on it circumference
    robot_map_prev = robot_map.copy()
    reveal_area = np.zeros(map.shape, np.uint8)

    reveal_area = cv2.circle(reveal_area, (pose[1], pose[0]), search_area, 255, -1)
    contours = np.argwhere(reveal_area==255)
    reveal_area = cv2.bitwise_and(map, reveal_area) #maybe unnecessary
    #Iterate drawing a line outwards to eachpixel in the area
    #For each line pixel, check if it's black, if it is every pixel after it us unknown
    for circ_point in contours:
        line = bresenham(pose[0], pose[1], circ_point[0], circ_point[1])
        for l_point in line:
            #If the pixel is currently unknown, set it to the value in the real map until a wall is reached 
            if robot_map_prev[l_point] == UNKNOWN:
                robot_map[l_point] = reveal_area[l_point]
            if reveal_area[l_point] == 0:
                break
    

                        
    return robot_map

#Bresenham line algorithm
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
        if e2 >= -dy:
            err -= dy
            x1 += sx
        if e2 <= dx:
            err += dx
            y1 += sy
            
    return points

# -- Frontier Handlers --
#Gets Edges of Unknown Areas
def get_frontier_points(robot_map, search_area):
    frontier_map = np.ones(np.shape(robot_map), dtype=np.uint8) * 255

    edge_map = cv2.bitwise_not(cv2.threshold(robot_map, UNKNOWN+1, 255, cv2.THRESH_BINARY)[1]) #map of edges of known area
    contours = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    wall_map = cv2.threshold(robot_map, UNKNOWN-1, 255, cv2.THRESH_BINARY)[1] #map of known walls
    frontier_map = cv2.drawContours(frontier_map, contours, -1, 0, 1) #Just edges of known area including walls

    frontier_map = cv2.bitwise_not(frontier_map) - cv2.bitwise_not(wall_map) #Edges of unknown space
    #Set image edges to 0 as they are wrongly identified as contours
    frontier_map[0, :] = 0
    frontier_map[:, 0] = 0
    frontier_map[np.shape(frontier_map)[0]-1, :] = 0
    frontier_map[:, np.shape(frontier_map)[1]-1] = 0

    #frontier_contours = cv2.findContours(frontier_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0] #may need to be cv2.bitwise_not()
    frontier_contours= np.argwhere(frontier_map==255)
    #frontier_contours = [cnt for cnt in frontier_contours if len(cnt) >= 5]
    
    #Displays frontier on robot map in red
    test_map = robot_map.copy()
    test_map = np.stack((test_map,) * 3, axis=-1)
    for point in frontier_contours:
         test_map[point[0], point[1]] = [0,0,255]
    #test_map = cv2.drawContours(test_map, frontier_contours, -1, [0,0,255],1)
    
    #f_points = np.unique(f_points)
    return frontier_contours, test_map

def frontier_cost(x, f_points, r_point, robot_map, search_area):
    #gains
    ak = 0.3
    dk = 1
    hk = 1
    #(f_points, r_point, robot_map, search_area) = args
    f_point = f_points[int(x[0])]

    f_point = adjust_frontier(f_point, cv2.threshold(robot_map, UNKNOWN+1, 255, cv2.THRESH_BINARY)[1])
   

    line = bresenham(r_point[0], r_point[1], f_point[0], f_point[1])
    
    if check_euclidian(line, robot_map) == True:
        
        path = line
        d = math.sqrt((f_point[0] - r_point[0])**2 + (f_point[1] - r_point[1])**2 )
    else:
        a_planner = a_star.PathPlanner(
            grid=list(1 + (-1 * cv2.threshold(robot_map, UNKNOWN+1, 255, cv2.THRESH_BINARY)[1].astype(float) / 255)), visual=False
        )
        path = a_planner.a_star(r_point[::-1], f_point[::-1])
        
        if path == -1:
            d = 1000000000000000000000
        else:
            path = [point[::-1] for point in path]
            d = np.sum(
                [
                    np.linalg.norm(np.array(y) - np.array(x))
                    for x, y in zip(path, path[1:])
                ]
            )
    reveal_area = np.zeros(robot_map.shape, np.uint8)
    reveal_area = cv2.circle(reveal_area, (f_point[1], f_point[0]), search_area, 255, -1)
    reveal_area = cv2.bitwise_and(robot_map, reveal_area)
    area_unknown = np.sum(reveal_area==UNKNOWN)

    hazards = 0
    if path != -1:
        for pos in path:
            if robot_map[pos[0], pos[1]] == HAZARD:
                        hazards = hazards + 1

    val = dk*d + hk*hazards - ak*area_unknown
    #print(val)
    return val

def eval_frontier(points, robot_pos, robot_map, search_radius):
    length = np.shape(points)[0]
    best_index = scipy.optimize.differential_evolution(
        func=frontier_cost,
        bounds=[(0, length-1)], 
        args=(points, robot_pos, robot_map, search_radius), 
        popsize=min(length//10, 20), 
        maxiter = 5,
        integrality=True
    )

    best_frontier = points[int(best_index.x[0])]
    best_frontier = adjust_frontier(best_frontier, cv2.threshold(robot_map, UNKNOWN+1, 255, cv2.THRESH_BINARY)[1])

    line = bresenham(robot_pos[0], robot_pos[1], best_frontier[0], best_frontier[1])
    if check_euclidian(line, robot_map) == True:
            path = line
    else:
        a_planner = a_star.PathPlanner(
            grid=list(1 + (-1 * cv2.threshold(robot_map, UNKNOWN+1, 255, cv2.THRESH_BINARY)[1].astype(float) / 255)), visual=False
        )
        path = a_planner.a_star(robot_pos[::-1], best_frontier[::-1])
        path = [point[::-1] for point in path]
    d = np.sum(
                    [
                        np.linalg.norm(np.array(y) - np.array(x))
                        for x, y in zip(path, path[1:])
                    ]
                )
    
    return best_frontier, path, d
     

def adjust_frontier(pose, map):
    if map[pose[0], pose[1]] == 0:
        if map[pose[0]-1, pose[1]] == 255:
            pose = (pose[0]-1, pose[1])
        elif map[pose[0]+1, pose[1]] == 255:
                pose = (pose[0]+1, pose[1])
        elif map[pose[0], pose[1]-1] == 255:
                pose = (pose[0], pose[1]-1)
        elif map[pose[0], pose[1]+1] == 255:
                pose = (pose[0], pose[1]+1)
    return pose

def check_euclidian(points, robot_map):
    for point in points:
        if robot_map[point] == 0 or (robot_map[point] == UNKNOWN and point != points[-1]):
            return False
    return True

# -- Main --
def main():
    st = time.time()

    total_dist = 0
    # -- Map Construction --
    Maps = [r"Maps/blank.png", r"Maps/blank_med.png", r"Maps/blank_small.png", r"Maps/hall.png", r"Maps/house.png", r"Maps/house_haz.png", r"Maps/Hall_haz.png"]
    Map = cv2.imread(Maps[1])
    Map_display = Map.copy() #Map Copy for display purposes
    Map = cv2.cvtColor(Map, cv2.COLOR_RGB2GRAY) #Map in grayscale for processing

    search_radius = 20

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
    start_pos = [249, 124]
    #start_pos = [249, 70]
    robot_pos = [start_pos]

    #robot_map = reveal_grid(robot_map, Map, robot_pos[-1], search_radius)
    while True:
        robot_map = reveal_grid(robot_map, Map, robot_pos[-1], search_radius)
        
        points, test_map = get_frontier_points(robot_map, search_radius)
        if len(points) == 0:
            break

        

        
        best_frontier, path, dist = eval_frontier(points, robot_pos[-1], robot_map, search_radius)
        total_dist += dist

        for point in robot_pos:
            test_map = cv2.circle(test_map, (point[1], point[0]), 3, [255,0,0], -1)
        for point in path:
             test_map[point[0], point[1]] = [255,0,0]
        cv2.imshow("points", test_map)
        cv2.waitKey(100) 

        #Avoids getting stuck in a wall - robot would usually avoid collision
        #best_frontier = adjust_frontier(best_frontier, Map)
        #point = adjust_frontier(point, Map)
        robot_pos.append(best_frontier)

    et = time.time() - st

    print(f"Exploration finished in {et:.2f}s and a total distance of {total_dist:.2f}")

if __name__ == "__main__":
    main()