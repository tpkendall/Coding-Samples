# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 21:50:04 2023

@author: thomas.kendall
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
"""
-------------------------------------------------------------------------------
Universals
-------------------------------------------------------------------------------
"""
speed_dic = {'walking': 1.4, 'crawling': .1, 'sneaking': .75, 'crawling_sneaking':.5, 'walking_sneaking':1} #VALIDATE THESE SPEEDS THROUGH TESTING!
height_dic = {'walking': 1.8, 'crawling': 1, 'sneaking': .15, 'crawling_sneaking':1, 'walking_sneaking':1.8} #CONFIRM HEIGHTS
transition_time_dic = {'crawling_sneaking': 2, 'walking_sneaking': .5} #CONFIRM THESE TIMES IN SECONDS!
seeker_orientation_uncertainty = {'human': (62*np.pi/180)+np.pi/2, 'bunker': 62*np.pi/180} #Enemy capabilities will influence this (among other things), will likely need to expand dictionary and definitions for each key
"""
-------------------------------------------------------------------------------
User Input
-------------------------------------------------------------------------------
"""
#change to appropriate file path for your computer
file_path="C:\\Users\\thomas.kendall\\OneDrive - West Point\\Documents\\Reference\\Research\\AY24-Research\\Optimal Tactical Route Project\\Jake\\OTP Figures\\" 
file_name=file_path+"Buckner" #map identifier
desired_lower_left_corner = (200, 200) #Given an image, assuming the bottom left corner is the origin, what is the bottom left corner of the map you want to look at
desired_upper_right_corner = (500, 500) #Given an image, assuming the bottom left corner is the origin, what is the bottom left corner of the map you want to look at
step_size=10  #desired distance, in meters, between nodes CANNOT BE TOO LARGE OR WILL CAUSE OverflowError when determining probability
seekers={1 : [(300/2,300/2), 15, 0, seeker_orientation_uncertainty['bunker']], 2 : [(150,275), 15, -np.pi/2, seeker_orientation_uncertainty['human']], 3 : [(50,50), 10, np.pi/4, seeker_orientation_uncertainty['bunker']]}
#{seeker ID number : [ (x,y), location uncertanty, orientation, orientation certainty ], next seeker : [...], ...}

"""
-------------------------------------------------------------------------------
Universal Calculations and Objects
-------------------------------------------------------------------------------
"""
elevation_map = Image.open(file_name+"_DEM.png") #read elevation map
vegetation_map = Image.open(file_name+"_NDVI.png") #read vegetation map
max_elevation=603 #max elevation on the map (CAN WE READ THIS FROM A CSV or .txt?)
map_width_meters=3000 #how wide is the map in meters (CAN WE READ THIS FROM A CSV or .txt?) CHANGE TO 800 FOR desert
map_length_meters=2999 #how long is the map in meters (CAN WE READ THIS FROM A CSV or .txt?) CHANGE TO 500 FOR desert
desired_map_width = desired_upper_right_corner[0]-desired_lower_left_corner[0] #Determine the desired map width
desired_map_length = desired_upper_right_corner[1]-desired_lower_left_corner[1] #Determine the desired map length
left, top, right, bottom = [desired_lower_left_corner[0],map_length_meters-desired_upper_right_corner[1],desired_upper_right_corner[0],map_length_meters-desired_lower_left_corner[1]] #translate into image coordinates (flipped y-axis)
map_width_pixels, map_length_pixels = elevation_map.size #Get image origional size in pixels
vegetation_map = vegetation_map.resize((map_width_pixels, map_length_pixels)) #Resize vegetation map to elevation map (should only be off by a few pixels)
crop_width_scale = map_width_pixels/map_width_meters #Conversion for scaling map x-limits in meters to map corners in pixels
crop_length_scale = map_length_pixels/map_length_meters #Conversion for scaling map y-limits in meters to map corners in pixels
(left, top, right, bottom) = (left*crop_width_scale, top*crop_length_scale, right*crop_width_scale, bottom*crop_length_scale) #Convert requested map edges in meters to pixel edges
elevation_map = elevation_map.crop((left, top, right, bottom)) #grab correct elevation map
vegetation_map = vegetation_map.crop((left, top, right, bottom)) #grab correct vegetation map
map_width_pixels, map_length_pixels = elevation_map.size #get size in pixels
map_width_scale=(map_width_pixels-1)/desired_map_width #calculate scaling factor for converting x coordinate to pixel coordinate
map_length_scale=(map_length_pixels-1)/desired_map_length #calculate scaling factor for converting y coordinate to pixel coordinate
nodes_wide=int(desired_map_length/step_size)+1 #how many node columns
nodes_long=int(desired_map_length/step_size)+1 #how many node rows

"""
-------------------------------------------------------------------------------
Visualization Functions
-------------------------------------------------------------------------------
"""
def detection_fields(mode_of_travel, perpendicular, plot_node_field, seekers=seekers):
    seeker_groups={templated_seeker : get_seeker_group(seekers[templated_seeker]) for templated_seeker in seekers}
    xvals=np.linspace(0, desired_map_width,2*nodes_wide-1)
    yvals=np.linspace(0, desired_map_length,2*nodes_long-1)
    travel_time = int(step_size/speed_dic[mode_of_travel])
    detection=[]
    # checked_locations=0
    for y in tqdm(yvals, desc="Progress", position=0, leave=True):
        detection.append([])
        for x in xvals:
            
            elevation= (elevation_map.getpixel((x*map_width_scale,y*map_length_scale))[0]/255)*max_elevation
            position_i=np.array([x,y, elevation])
            
            visual_detection = get_visual_detection_2(position_i, mode_of_travel, travel_time, seeker_groups, perpendicular)
            audio_detection = get_audio_detection(position_i, position_i, mode_of_travel, seeker_groups) #WILL NEED TO FIX
            # checked_locations+=1
            # print(progress_bar(np.round(checked_locations/total_locations, 2)))
            detection[-1].append(max(visual_detection,audio_detection))
    plt.clf()
    plt.style.use('ggplot')
    fig, ax= plt.subplots()
    im = ax.imshow(detection, extent=[0, xvals[-1], 0, yvals[-1]],
                    origin='lower', cmap='viridis')    
    # cb_ax = fig.add_axes([0.83, 0.3, 0.02, 0.4])
    # fig.colorbar(im, cax=cb_ax, label="Detection Probability")
    
    for seeker in seekers:
        [(seeker_x,seeker_y), z, seeker_orient, seeker_orient_uncertainty] = seekers[seeker]
        ax.arrow(seeker_x, seeker_y, 2*step_size*np.cos(seeker_orient), 2*step_size*np.sin(seeker_orient), width=step_size/10, head_width=step_size/2, color='red')
        thetas=np.linspace(0, 2*np.pi,100)
        xs = [seeker_x+z*np.cos(thetas[i]) for i in range(100)]
        ys = [seeker_y+z*np.sin(thetas[i]) for i in range(100)]
        ax.plot(xs,ys,color='red')
    fig.colorbar(im, label="Detection Probability")
    if plot_node_field:
        node_field=create_node_field()
        for i in range(nodes_wide*nodes_long):
            (x,y)=node_field[i+1][0]
            plt.scatter(x,y,color="black")
    return

def get_visual_detection_2(position_i, mode_of_travel, travel_time, seeker_groups, perpendicular):

    for seeker in seekers:
        [seeker_coord, z, orient, orient_uncert] = seekers[seeker]
        distance_position_i=np.linalg.norm(np.array(seeker_coord)-position_i[:2])
        if distance_position_i <=z*np.sqrt(2):
            return 1
    visual_detection=[]
    for worst_case_seekers in seeker_groups:

        for seeker in seeker_groups[worst_case_seekers]:
            los = get_los(seeker, position_i) #Line of Sight to start postion
            
            if los==0:
                visual_detection.append(0)
            else:
                [seeker_x, seeker_y, seeker_elevation, orient_left, orient_right] = seeker
                seeker_coord = np.array([seeker_x,seeker_y,seeker_elevation])
                seeker_to_evader = position_i-seeker_coord
                if perpendicular:
                    evader_v = rotate(np.pi/2) @ seeker_to_evader[:2]
                    evader_v = evader_v / np.linalg.norm(evader_v) #move perpendicular to seekers direct line of sight
                    position_j = position_i + speed_dic[mode_of_travel]*np.append(evader_v, position_i[-1])
                else:
                    evader_v = seeker_to_evader[:2] / np.linalg.norm(seeker_to_evader[:2])
                    position_j = position_i + speed_dic[mode_of_travel]*np.append(evader_v, position_i[-1])
                alpha = get_alpha(seeker_coord, position_i, position_j, speed_dic[mode_of_travel])
                beta = get_beta(seeker_coord, position_i, position_j, height_dic[mode_of_travel])
                trace_ratio = closed_form_ratio(alpha, beta)
                # detection_probability = 999*trace_ratio/(998*trace_ratio+1) #THIS FUNCTION COULD USE SOME UPDATING!
                detection_probability = 101*trace_ratio/(100*trace_ratio+1) #THIS FUNCTION COULD USE SOME UPDATING!
                detection_probability = detection_over_step(int(travel_time), detection_probability) #testing this function out
                visual_detection.append(los*detection_probability)
    return max(visual_detection)

def visualize_node_field(requested_mode, plot_nodes):
    
    colors_dic={'walking': 'red', 'crawling': 'blue', 'sneaking': 'green', 'crawling_sneaking':'black', 'walking_sneaking':'black'}
    
    arcs=get_arcs()
    plotted_arcs=[]
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for arc in arcs:
        (node_i, node_j) = arc
        reverse_arc = (node_j, node_i)
        if reverse_arc not in plotted_arcs:
            [ (x_i, y_i, z_i), (x_j, y_j, z_j), mode_of_travel, time, risk ] = arcs[arc]
            # print(z_i, z_j)
            if mode_of_travel == requested_mode or requested_mode=='all':
                if plot_nodes:
                    ax.scatter(x_i, y_i, z_i, color='black')
                    ax.scatter(x_j, y_j, z_j, color='black')
                plt.plot([x_i,x_j],[y_i,y_j],[z_i,z_j],color=colors_dic[mode_of_travel])
                plotted_arcs.append(arc)
                plotted_arcs.append(reverse_arc)
    custom_lines = [Line2D([0], [0], color='red', lw=1),
                Line2D([0], [0], color='blue', lw=1),
                Line2D([0], [0], color='green', lw=1),
                Line2D([0], [0], color='black', lw=1)]
    plt.legend(custom_lines, ['walking', 'crawling', 'sneaking', 'chaning'])
    return
        
"""
-------------------------------------------------------------------------------
Main Functions
-------------------------------------------------------------------------------
"""
def get_arcs(nodes_wide=nodes_wide, nodes_long=nodes_long, step_size=step_size, file_name=file_name, max_elevation=max_elevation, seekers=seekers):
    """
    gets arcs and node field

    Parameters
    ----------
    nodes_wide : Integer
        width of node field measured in nodes.
    nodes_long : Integer
        length of node field measured in nodes.
    step_size : Float
        desired birds eye distance between N-S or E-W adjacent nodes.
    file_name : String
        Name of map: typically changed universally as 'map_name_location'
    max_elevation : Float
        Maximum elevation on elevation imagery.
    seekers : dictionary
        { seeker ID number : [ (x,y), location uncertanty, orientation, orientation certainty ]}. NEED TO ADD CAPABILITIES

    Returns
    -------
    arcs : Dictionary
        Keys are tuples that indicate the arcs start and end nodes e.g. (i,j).
        definitions are node coordinates (including elevation), mode of travel, travel time, and probabiliy of detection (NEED TO ADD FUEL CONSUMPTION)
        e.g. {(node i, node j) : [ node_i location (x,y,z), node_j location (x,y,z), mode of travel, time, detection probability ], ... }
    
    """
    # start_time = time.time()
    
    node_field=create_node_field()
    
    seeker_groups={templated_seeker : get_seeker_group(seekers[templated_seeker]) for templated_seeker in seekers}
    
    single_field=nodes_wide*nodes_long
    crawling_nodes=[i+1 for i in range(single_field)]
    sneaking_nodes=[i+1+single_field for i in range(single_field)]
    walking_nodes=[i+1+2*single_field for i in range(single_field)]
    
    "Create arcs dictionary"
    arcs={}
    # arc_length=get_arc_length()/2
    N=3*single_field

    checked=0
    for i in tqdm(range(N)):
        node_i=i+1
        [coordinate_i, elevation_i, vegetation_i, adjacent_nodes_i] = node_field[node_i]
        position_i=np.array(coordinate_i+(elevation_i,))
        node_i_land=classify_node(position_i)
        for node_j in adjacent_nodes_i:
            "Dont recaclulate arcs arleady found"
            key=(node_j,node_i)
            if key not in arcs:
                [coordinate_j, elevation_j, vegetation_j, adjacent_nodes_j] = node_field[node_j]
                position_j=np.array(coordinate_j+(elevation_j,))
                node_j_land=classify_node(position_j)
                
                "Only add Arc if arc is possible (no cliff and neither node is in water)"
                if abs(elevation_i-elevation_j)<=2*step_size and node_i_land and node_j_land: #WHAT IS THE STEEPEST GRADE A HUMAN CAN CLIMB
                    
                    "Determine Time for travel"
                    if coordinate_i == coordinate_j:
                        if (node_i in crawling_nodes and node_j in sneaking_nodes) or (node_j in crawling_nodes and node_i in sneaking_nodes):
                            mode_of_travel='crawling_sneaking'
                            
                        else:
                            mode_of_travel='walking_sneaking'
                        travel_time=transition_time_dic[mode_of_travel]
                    else:
                        distance = np.linalg.norm(position_j-position_i)
                        if node_i in walking_nodes:
                            mode_of_travel='walking'
                        elif node_i in sneaking_nodes:
                            mode_of_travel='sneaking'
                        else:
                            mode_of_travel='crawling'
                        
                        "Determine time base on speed, and average vegetation encountered"
                        average_vegetation=(vegetation_i+vegetation_j)/2
                        vegetation_factor=1-(2/9)*average_vegetation #NEEDS VALIDATION THROUGH TESTING!!!!!!!!!!!
                        travel_time=distance/(vegetation_factor*speed_dic[mode_of_travel])
                    
                    "Determine Probability of detection"
                    visual_detection=get_visual_detection(position_i, position_j, mode_of_travel, travel_time, seeker_groups)
                    audio_detection=get_audio_detection(position_i, position_j, mode_of_travel, seeker_groups)
                risk_level = max(visual_detection,audio_detection)    
                arcs[(node_i,node_j)]=[position_i, position_j, mode_of_travel, travel_time, risk_level]
                arcs[(node_j,node_i)]=[position_j, position_i, mode_of_travel, travel_time, risk_level]
                checked+=1
                # print(node_i,"-> ", node_j,"complete, risk=", np.round(risk_level,4), "Status:",np.round(100*(checked)/arc_length,2),"% Complete")
    # print("Calculation Time:",time.time()-start_time)
    return arcs

def create_node_field():
    """
    Creates node field

    Parameters (All Universal)
    ----------
    nodes_wide : Integer
        width of node field measured in nodes.
    nodes_long : Integer
        length of node field measured in nodes.
    step_size : Float
        desired birds eye distance between N-S or E-W adjacent nodes.
    file_name : String
        Name of map: typically changed universally as 'map_name_location'
    max_elevation : Float
        Maximum elevation on elevation imagery.

    Returns
    -------
    node_field : Dictionary
        Keys are node id numbers (1,...,3wl) and definitions are coordinates (as a tuple), elevation, and adjacent nodes (as a list)
        e.g. {Node Id Number : (x,y), elevation, vegetation, [adjacent nodes], ... }.

    """
    node_field={}
    single_field=nodes_wide*nodes_long

    node_id=1
    for l in range(nodes_long):
        for w in range(nodes_wide):
            coordinate = (w*step_size,l*step_size)
            (x,y)=(coordinate[0]*map_width_scale,coordinate[1]*map_length_scale)
            elevation = (elevation_map.getpixel((x, y))[0]/255)*max_elevation
            r= vegetation_map.getpixel((x, y))[0] 
            vegetation = (3-(3*(r/255))) #vegetation is scaled "continuously" from 0 (None) to 3 (Dense)
            #crawling level
            node_field[node_id] = [coordinate, elevation+.15, vegetation, get_adjacent_nodes(node_id, coordinate)]
            
            #sneaking level
            node_field[node_id+single_field] = [coordinate, elevation+1, vegetation, get_adjacent_nodes(node_id+single_field, coordinate)]
            
            #walking level
            node_field[node_id+2*single_field] = [coordinate, elevation+1.8, vegetation, get_adjacent_nodes(node_id+2*single_field, coordinate)]
            node_id+=1
    return node_field
        
def get_adjacent_nodes(node_id, coordinate):
    """
    Gets the adjacent nodes for a specific node number

    Parameters
    ----------
    node_id : Integer
        The node which you wish to know its adjacent nodes.
    coordinate : tuple
        x and y coordinates of node_id.
    nodes_wide : Integer (UNIVERSAL)
        width of node field measured in nodes.
    nodes_long : Integer (UNIVERSAL)
        length of node field measured in nodes.
    step_size : Float (UNIVERSAL)
        desired birds eye distance between N-S or E-W adjacent nodes.

    Returns
    -------
    actual_adjacents : list
        list of node ID numbers that are adjacent to the given node_id

    """
    single_field=nodes_wide*nodes_long
    potential_adjacents=[node_id+1,node_id+nodes_wide+1,node_id+nodes_wide,
                         node_id+nodes_wide-1,node_id-1,node_id-nodes_wide-1, 
                         node_id-nodes_wide,node_id-nodes_wide+1, 
                         node_id+single_field,node_id-single_field] #Possible nodes starting at theta=0 and proceeding pi/4 and then adding next level and previous level
    potential_adjacents_locations=[(coordinate[0]+step_size,coordinate[1]),(coordinate[0]+step_size,coordinate[1]+step_size),(coordinate[0],coordinate[1]+step_size),
                                   (coordinate[0]-step_size,coordinate[1]+step_size),(coordinate[0]-step_size,coordinate[1]),(coordinate[0]-step_size,coordinate[1]-step_size),
                                   (coordinate[0],coordinate[1]-step_size),(coordinate[0]+step_size,coordinate[1]-step_size),coordinate,coordinate] #Possible node locations starting at theta=0 and proceeding pi/4 and then adding next level and previous level
    actual_adjacents=[]

    for i in range(10):
        potential_location=potential_adjacents_locations[i]
        in_horizon = potential_adjacents[i]>0 and potential_adjacents[i]<=3*single_field
        in_map = potential_location[0]>=0 and potential_location[0]<step_size*nodes_wide and potential_location[1]>=0 and potential_location[1]<step_size*nodes_long
        if in_horizon and in_map:
            actual_adjacents.append(potential_adjacents[i])

            
    return actual_adjacents

def get_visual_detection(position_i, position_j, mode_of_travel, travel_time, seeker_groups):
    """
    This function should get the probability of visual detection
    Tasks:
        o Determine line of sight to start and end positions (seperate function that checks if evader is in deadspace or is blocked by foliage)
        o Do the necessary calculus (Jakes function)
    """
    
    for seeker in seekers:
        [seeker_coord, z, orient, orient_uncert] = seekers[seeker]
        distance_position_i=np.linalg.norm(np.array(seeker_coord)-position_i[:2])
        distance_position_j=np.linalg.norm(np.array(seeker_coord)-position_j[:2])
        if distance_position_i <=z or distance_position_j <= z:
            return 1

    visual_detection=[]
    for worst_case_seekers in seeker_groups:

        for seeker in seeker_groups[worst_case_seekers]:
            los_i = get_los(seeker, position_i) #Line of Sight to start postion
            los_j = get_los(seeker, position_j) #Line of sight to stop position
            los = (los_i+los_j)/2 #average line of sight from start to stop
            
            if los==0:
                visual_detection.append(0)
            else:
                [seeker_x, seeker_y, seeker_elevation, orient_left, orient_right] = seeker
                seeker_coord = np.array([seeker_x,seeker_y,seeker_elevation])
                alpha = get_alpha(seeker_coord, position_i, position_j, speed_dic[mode_of_travel])
                beta = get_beta(seeker_coord, position_i, position_j, height_dic[mode_of_travel])
                trace_ratio = closed_form_ratio(alpha, beta)
                # detection_probability = 999*trace_ratio/(998*trace_ratio+1) #THIS FUNCTION COULD USE SOME UPDATING!
                detection_probability = 101*trace_ratio/(100*trace_ratio+1) #THIS FUNCTION COULD USE SOME UPDATING!
                detection_probability = detection_over_step(int(travel_time), detection_probability) #testing this function out
                visual_detection.append(los*detection_probability)
    return max(visual_detection)

def get_los(seeker, evader_loc):
    if evader_in_blindspot(seeker, evader_loc):
        # print("blindspot")
        return 0
    [seeker_x, seeker_y, seeker_elevation, orient_left, orient_right] = seeker
    seeker_loc=np.array([seeker_x, seeker_y, seeker_elevation])    
    
    "Create vector function r(t) from seeker to evader (the seeker's line of sight)"
    r0=seeker_loc[:]
    v=evader_loc-seeker_loc
    distance=np.linalg.norm(v)
    t=np.linspace(0,1,int(distance/(step_size/2.5))) #searches along route every ~step_size/2.5 meters
    r=[r0+v*t[i] for i in range(len(t))]
    
    "Find the vegetation factor on visibility"
    vegetation_factor=1
    for position in r:
        [x,y,e]=position
        ground_elevation = (elevation_map.getpixel((x*map_width_scale,y*map_length_scale))[0]/255)*max_elevation
        if ground_elevation>e:
            "Line of sight blocked by obstace: Evader is in deadspace"
            # print("deadspace")
            return 0
        r = vegetation_map.getpixel((x*map_width_scale,y*map_length_scale))[0]
        vegetation = (3-(3*(r/255)))
        vegetation_factor *= (1-(1/30)*vegetation) #assumes linear probability of seeing through vegetation probability = 1-(2/30)*density at any given point.
        #instances of seeing through vegetatin are independent, thus the probability of seeing through a bunch of vegetation of their multiplication.
        if vegetation_factor<.01:
            # print("VEGETATED")
            return 0
    return vegetation_factor
    

def get_audio_detection(position_i, position_j, mode_of_travel, seeker_groups):
    """
    This function should get the probability of visual detection
    Tasks:
        o Do the necessary calculations (Jakes function)
    """
    return 0

def audio_f(distance, mode_of_travel, vegetation):
    """
    Returns audio probability of detection of evader at node

    pointsource is the decibal level at mode of movement and vegetation level

    Parameters
    ----------
    dist : Float. Distance to evader from nearest seeker.
    evader_pos : String. evader's mode of movement (w,s,c)
    veg: Integer. vegetation score at evader's position
    """
    if mode_of_travel == 'walking':
        point_source = (20/3)*vegetation+17

    elif mode_of_travel == 'sneaking':
        point_source = (14/3)*vegetation+6

    elif mode_of_travel == 'crawling':
        point_source = (6/3)*vegetation+2

    a = probability_audio(point_source, distance)

    return a

# p is the decibal level of moving through vegetation level

# flip the y values of the entire array and imagery

def probability_audio(point_source, dist):
    """
    Parameters
    ----------
    point_source : TYPE
        this is the decibal level of the evaders movement, dependent on the type of vegetation
    dist : TYPE
        DESCRIPTION.

    Returns
    -------
    final : TYPE
        DESCRIPTION.

    """
    r = dist
    exponent_1 = -(point_source-4)/20
    exponent_2 = 10*np.exp(exponent_1)*(r-np.exp(-exponent_1))
    final = 1/(1 + np.exp(exponent_2))
    return final
"""
-------------------------------------------------------------------------------
Supporting Functions
-------------------------------------------------------------------------------
"""    
    
def get_arc_length():
    
    return 12 - 18 * (nodes_wide+nodes_long)+28 * nodes_wide * nodes_long

def get_seeker_group(templated_seeker):
    """
    takes a templated seeker location and creates a list of worst case seeker information.
    Assumes seeker is 2m tall

    Parameters
    ----------
    templated_seeker : list
        [seeker_loc, loc_uncertainty, theta_orientation, theta_uncertainty].
    node_field_info : list
        [nodes_wide, nodes_long, step_size, file_name, max_elevation, node_field].

    Returns
    -------
    seeker_group : list of lists
        [ worst case seeker 1 info (as a list of x coordinate, y coordinate, elevation, left limit to vision, right limit to vision),
         worst case seeker 1 info, 
         ...,
         worst case seeker max elevation info].

    """
    [seeker_loc, loc_uncertainty, theta_orientation, theta_uncertainty] = templated_seeker
    seeker_loc=np.array(seeker_loc)
    seeker_box = [seeker_loc+loc_uncertainty*rotate(theta_orientation)@np.array([(-1)**i,(-1)**j]) for i in range(2) for j in range(2)] #Get locations of corner seekers
    orient_left=theta_orientation+theta_uncertainty
    orient_right=theta_orientation-theta_uncertainty
    "Create locations of corner seekers listed as (x,y,elevation)"
    seeker_group=[]
    for i in range(4):
        [x,y] = seeker_box[i]
        seeker_group.append([x, y, (elevation_map.getpixel((x*map_width_scale,y*map_length_scale))[0]/255)*max_elevation+2, orient_left, orient_right])
        
    [x_center,y_center]=seeker_loc
    locations=[[(x_center-loc_uncertainty+i,y_center-loc_uncertainty+j) for i in range(2*loc_uncertainty)] for j in range(2*loc_uncertainty)]

    "Find location of highest seeker in seeker box and add to seeker box"
    e_max=-999
    for i in range(2*loc_uncertainty):
        for j in range(2*loc_uncertainty):
            [x2,y2] = locations[i][j]
            e=(elevation_map.getpixel((x2*map_width_scale,y2*map_length_scale))[0]/255)*max_elevation
            if e>e_max:
                e_max=e
                [loc_x,loc_y]=locations[i][j]
    max_elevation_seeker = [loc_x, loc_y, e_max+2, orient_left, orient_right]
    seeker_group.append(max_elevation_seeker)
    
    return seeker_group

def rotate(theta):
    """
    2D rotational matrix to rotate 2D coordinates theta radians

    Parameters
    ----------
    theta : float
        desired rotation in radians.

    Returns
    -------
    R_theta : numpy array
        rotational matrix.

    """
    R_theta=np.array([[np.cos(theta),-np.sin(theta)],
                      [np.sin(theta),np.cos(theta)]])
    return R_theta

def evader_in_blindspot(seeker, evader_loc):
    [seeker_x, seeker_y, seeker_elevation, orient_left, orient_right] = seeker
    ll = pos_angle(orient_left)
    rl = pos_angle(orient_right)
    seeker=np.array([seeker_x,seeker_y])
    evader=evader_loc[:2]
    s_e=evader-seeker
    
    angle_to_evader = np.arctan2(s_e[1],s_e[0])
    if true_range_angle(angle_to_evader, ll, rl):
        return True
    
    return False
    
def pos_angle(angle):
    """
    Takes any angle and returns that angle mapped to [0,2pi]
    """
    if angle < 0:
        return pos_angle(angle+2*np.pi)
    elif angle > 2*np.pi:
        return pos_angle(angle-2*np.pi)
    return angle

def true_range_angle(alpha, angle1, angle2):
    """
    Calculates if an angle is between two angles. Returns Boolean.
    """
    alpha = pos_angle(alpha)
    angle2 = pos_angle(angle2-angle1)
    alpha = pos_angle(alpha-angle1)

    if alpha < angle2:
        return True
    return False

def classify_node(node_coordinate):
    "This function should check if node is land (True) or water (False)"
    return True

def get_alpha(seeker_node, start_node, end_node, speed):
    """
    Get the birds eye angle between seeker to start and seeker to end.

    Parameters
    ----------
    seeker_node : numpy array
        coordinates of the seeker (x, y, elevation).
    start_node : numpy array
        starting coordinates of the evader (x, y, elevation).
    end_node : numpy array
        ending coordinates of the evader (x, y, elevation).
    speed : float
        evader's maximum possible (read worst case) speed in m/s.

    Returns
    -------
    alpha : float
        birds eye angle between seeker to start and seeker to end (in radians).

    """
    n_1 = np.array(start_node[:2])
    n_2 = np.array(end_node[:2])
    s = np.array(seeker_node[:2])
    if np.array_equiv(n_1, n_2):
        "In this case the person is transitioning modes and alpha is the worst case width of the evader"
        s_perpendicular=rotate(np.pi/2)@s
        s_perpendicular_unit = s_perpendicular / np.linalg.norm(s_perpendicular)
        n_2=n_1+s_perpendicular_unit*.5
   
    v = n_2 - n_1
    a = n_1 - s
    b = a + speed * v/np.linalg.norm(v)
    a_unit= a / np.linalg.norm(a)
    b_unit= b / np.linalg.norm(b)
    alpha = np.arccos(a_unit @ b_unit)
    return alpha


def get_beta(seeker_node, start_node, end_node, height):
    """
    Gets vertical angle use to calculate trace.

    Parameters
    ----------
    seeker_node : numpy array
        coordinates of the seeker (x, y, elevation).
    start_node : numpy array
        starting coordinates of the evader (x, y, elevation).
    end_node : numpy array
        ending coordinates of the evader (x, y, elevation).
    height : float
        height of the evader (tied to movement mode).

    Returns
    -------
    beta : float
        worst case vertical angle trace of evader during movement.

    """
    maximum_height=max(start_node[-1],end_node[-1])
    minimum_height=min(start_node[-1]-height,end_node[-1]-height)
    node_sorter=[start_node, end_node]
    distances=[np.linalg.norm(seeker_node-start_node), np.linalg.norm(seeker_node-end_node)]
    [x_start, y_start, elevation_start] = node_sorter[np.argmin(distances)]
    n1 = np.array([x_start, y_start, minimum_height])
    n2 = np.array([x_start, y_start, maximum_height])
    a = n1 - seeker_node
    b = n2 - seeker_node
    a_unit= a / np.sqrt(a @ a)
    b_unit= b / np.sqrt(b @ b)
    beta = np.arccos(a_unit @ b_unit)
    return beta

def closed_form_ratio(alpha, beta):
    evader_trace_1 = 2 * np.pi
    evader_trace_2 = np.arcsin(
        ((np.cos(alpha / 2)) * np.tan(beta / 2)) / np.sqrt((np.tan(alpha / 2) ** 2) + (np.tan(beta / 2) ** 2)))
    evader_trace_3 = np.arcsin(
        ((np.cos(beta / 2)) * np.tan(alpha / 2)) / np.sqrt((np.tan(alpha / 2) ** 2) + (np.tan(beta / 2) ** 2)))
    evader_trace = evader_trace_1 - 4 * (evader_trace_2 + evader_trace_3)
    seeker_visual_field = 3.7505
    closed_form = evader_trace / seeker_visual_field
    return min(1, closed_form)

def detection_over_step(steps, probability_single_step):
    # start_time=time.time()
    total_probability=sum([((-1)**i)*math.comb(steps,i+1)*(probability_single_step**(i+1)) for i in range(steps)])
    # print(time.time()-start_time)
    return total_probability  

"""
-------------------------------------------------------------------------------
Ideas to Implement
-------------------------------------------------------------------------------
    Increase Speed    
        o Instead of a seeker box of half length z, create a seeker circle of radious z (different type of uncertainty)
            - Find true angle (theta) from seeker to evader (arctan2)
            - seeker_closest= rotate(theta) @ (seeker+(z,0))
            - seeker_left = rotate(theta) @ (seeker+(0, z))
            - seeker_right = rotate(theta) @ (seeker+(0, -z))
            - seeker_max = max elevation seeker (grid / ?polar? search in circle of uncertainty)
        o Only check three closest seekers in each seeker box
        o Analyze f(gamma) as a function of distance and angle between movement and evader to seeker
            - Potentially remove need to calculate alpha beta and gamma
        o 
"""