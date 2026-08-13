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

    frontier_contours = cv2.findContours(frontier_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0] #may need to be cv2.bitwise_not()
    
    
    frontier_contours = [cnt for cnt in frontier_contours if len(cnt) >= 5]
    
    #Displays frontier on robot map in red
    test_map = robot_map.copy()
    test_map = np.stack((test_map,) * 3, axis=-1)
    test_map = cv2.drawContours(test_map, frontier_contours, -1, [0,255,0],1)

    f_points = []
    #Get frontier point candidates to evaluate from centroids of contours
    for f_contour in frontier_contours:
        f_contour = np.squeeze(f_contour)
        con_len = np.shape(f_contour)[0]
        num_sections = con_len//search_area
        if num_sections >= 1:
            sections = np.array_split(f_contour, num_sections)
            for section in sections:
                midpoint = section.shape[0]//2
                f_points.append([int(section[midpoint,1]),int(section[midpoint,0])])
        else:
            section = f_contour
            midpoint = section.shape[0]//2
            f_points.append([int(section[midpoint,1]),int(section[midpoint,0])])
       

    '''
    for f_contour in frontier_contours:
        
        #print(f_contour) #Array of 2D arrays [[x y]] 
        if np.round(np.shape(f_contour)[0]/(search_area))/2 > 1:
            num_sections = np.round(np.shape(f_contour)[0] / (search_area))/2
            print(num_sections)
            sections = np.array_split(f_contour, num_sections)
        else:
            sections = f_contour
        for section in sections:
            
            length = np.shape(section)[0]
            x = np.empty(length)
            y = np.empty(length)
            i = 0
            for point in section:
                point = point.squeeze()

                x[i] = point[1]
                y[i] = point[0]
                i = i+1
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            f_points.append(np.asanyarray([np.round(sum_x/length), np.round(sum_y/length)]).astype(np.int32))
            
            print(section)
            f_points.append(section[section.size()//2])
    '''
    #print(f_points)


    test_map = robot_map.copy()
    test_map = np.stack((test_map,) * 3, axis=-1)
    test_map = cv2.drawContours(test_map, frontier_contours, -1, [0,0,255],1)
    for point in f_points:
        test_map = cv2.circle(test_map, (point[1], point[0]), 2, [0,0,255], -1)
       

    #f_points = np.unique(f_points)
    return f_points, test_map

def evaluate_frontier_points(points, robot_pose, robot_map, Map, search_area):
    #gains
    ak = 0.3
    dk = 1
    hk = 1

    best = 1000000000000000000000
    reveal_area = np.zeros(robot_map.shape, np.uint8)
    #reveal_area = cv2.circle(reveal_area, (robot_pose[1], robot_pose[0]), search_area, 255, -1)
    #area_total = np.count_nonzero(reveal_area)

    
    for point in points:
        #Euclidian Distance
        point = adjust_frontier(point, cv2.threshold(robot_map, UNKNOWN+1, 255, cv2.THRESH_BINARY)[1])
        line = bresenham(robot_pose[0], robot_pose[1], point[0], point[1])
        if check_euclidian(line, cv2.threshold(robot_map, UNKNOWN+1, 255, cv2.THRESH_BINARY)[1]) == True:
            path = line
            d = math.sqrt((point[0] - robot_pose[0])**2 + (point[1] - robot_pose[1])**2 )
        else:
            a_planner = a_star.PathPlanner(
                grid=list(1 + (-1 * cv2.threshold(robot_map, UNKNOWN+1, 255, cv2.THRESH_BINARY)[1].astype(float) / 255)), visual=False
            )
            path = a_planner.a_star(robot_pose[::-1], point[::-1])
            
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


        #New Info Calc
        reveal_area = cv2.circle(reveal_area, (point[1], point[0]), search_area, 255, -1)
        reveal_area = cv2.bitwise_and(robot_map, reveal_area)
        area_unknown = np.sum(reveal_area==UNKNOWN)

        #Hazard Calc
        #path = bresenham(robot_pose[0], point[0], robot_pose[1], point[1]) #needs a better method - not guarenteed to be a straight line 
        hazards = 0
        if path != -1:
            for pos in path:
                if robot_map[pos[0], pos[1]] == HAZARD:
                    hazards = hazards + 1

        val = dk*d + hk*hazards - ak*area_unknown
        if val < best:
            best = val
            best_pos=point
            best_path = path
            best_dist = d

    return best_pos, best_path, best_dist

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
        if robot_map[point] == 0:# or (robot_map[point] == UNKNOWN and point != points[-1]):
            return False
    return True

# -- Main --
def main():
    st = time.time()

    total_dist = 0
    # -- Map Construction --
    Maps = [r"Maps/blank.png", r"Maps/blank_med.png", r"Maps/blank_small.png", r"Maps/hall.png", r"Maps/house.png", r"Maps/house_haz.png", r"Maps/Hall_haz.png"]
    Map = cv2.imread(Maps[0])
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

        best_frontier, path, dist = evaluate_frontier_points(points, robot_pos[-1], robot_map, Map, search_radius)
        total_dist = total_dist + dist

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