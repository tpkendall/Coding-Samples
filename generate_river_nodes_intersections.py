# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:10:01 2023

@author: thomas.kendall
"""

import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import csv
import os

#Univerals
node_type={"water":.1,"rail":.05,"air":.01,"intersection":0}
key_sort=["water","rail","air","intersection"]
L=len(key_sort)

"""
This code was created to simulate multimodal transportation networks to practice optmizing over.

Try generate_scenarios(200,200,1000,1) to generate a single scenario 200x200 mile region with 1000 randomly generated intersections and a single randomly generated river

Once this is done, feel free to access the same information by reading the csv outputs. You will need to update FILEPATH to the location you ran this code.
    o If looking to see scenario_i (for any given i, width w, length l, and node co), without seeing air connections, 
      call plot_scenario(get_matrices("FILEPATH\\Scenarios\\scenario_i\\",w,l,n), False)
    o If you have a set of solution paths=[path1, path2, ... pathN] where each path is a collection of node id numbers
      call path_scenario_overlay(paths,FILEPATH\\Scenarios\\scenario_i\\",w,l,n)
    o If you want to highlight a specific node or set of nodes call highlight_node(get_matrices("Scenarios\\Approved Scenarios\\scenario_i\\",w,l,n),requested_node)
    
"""

"""
---------------------------------------------------------------------------------------------------------------
For reading potential node fields to optimize over
---------------------------------------------------------------------------------------------------------------
"""

def get_start_nodes(filepath_name, width=200, length=200, nodes=1000):
    A, D, river, desc, node_loc, width, length, nodes=get_matrices(filepath_name, width, length, nodes)
    opp_dic={1:3,2:4,3:1,4:2}
    quad_dic={1:[150,150], 2: [0,150], 3:[0,0], 4:[150,0]}
    start_quadrant=rnd.randint(1, 4)
    end_quadrant=opp_dic[start_quadrant]
    [start_left, start_bottom] = quad_dic[start_quadrant]
    [end_left, end_bottom] = quad_dic[end_quadrant]
    x_start=start_left+50*rnd.random()
    y_start=start_bottom+50*rnd.random()
    x_end=end_left+50*rnd.random()
    y_end=end_bottom+50*rnd.random()
    start=np.array([x_start,y_start])
    end=np.array([x_end,y_end])
    distances_to_start=[]
    distances_to_end=[]
    for row in node_loc:
        [node_id,x,y,w,r,a]=row
        node_coordinate=np.array([x,y])
        distances_to_start.append(np.linalg.norm(start-node_coordinate))
        distances_to_end.append(np.linalg.norm(end-node_coordinate))
    descriptions_list=["Number of intersections: ","Number of water nodes: ",
                        "Number of rail nodes: ","Number of air nodes: ", 
                        "Number of river nodes: "]
    for i in range(len(descriptions_list)):
        descriptions_list[i]+=str(desc[0][i])
    descriptions_list.append("Start node ID: "+str(np.argmin(distances_to_start)+1))
    descriptions_list.append("End node ID: "+str(np.argmin(distances_to_end)+1))
    with open(filepath_name+"info.txt",'w') as f:
        for line in descriptions_list:
            f.write(line)
            f.write('\n')
    return 
        
def generate_info_docs(num_scenarios,width=200,length=200,nodes=1000):
    for scenario in range(num_scenarios):
        filepath="Scenarios\\Approved Scenarios\\scenario_"+str(scenario+1)+"\\"
        get_start_nodes(filepath, width, length, nodes)
    return

def path_scenario_overlay(paths,filepath_name, width, length, nodes):
    """
    Overlays paths over the node network.

    Parameters
    ----------
    paths : list
        list of paths. MUST BE A LIST EVEN IF ONLY ONE PATH IS IN PATHS. ie paths=[path].
    filepath_name : str
        file path to .
    width : int
        how many miles wide is the the scenario (check file names in scenario folder).
    length : int
        how many miles long the is the scenario (check file names in scenario folder).
    nodes : int
        number of intersections in the scenario (check file names in scenario folder).
    """
    plt.clf()
    "read in csvs"
    adjacency, distance, river, desc, node_loc, width, length, nodes=get_matrices(filepath_name, width, length, nodes)
    "plot nodes and connections"
    plot_scenario([adjacency, distance, river, desc, node_loc, width, length, nodes],False)
    i=0
    number_of_paths=len(paths)
    for path in paths:
        path_name="Path "+str(i+1)
        plot_path(path, node_loc, width, length, path_name,i,number_of_paths)
        i+=1
    return

def update_plot_nodes(node,node_loc,d,th,desc):
    [N,W,R,A,Riv,T]=desc
    theta=np.linspace(0,2*np.pi,180)

    for row in node_loc:
        # print(row)
        [node_id,x,y,w,r,a]=row
        # print(node_id)
        if node_id-1<N:
            plt.scatter(x,y,color="black",marker="o")
        elif node_id-1<N+W:
            x+=d
            plt.scatter(x, y, color="blue",marker="s")
        elif node_id-1<N+W+R:
            x+=d*np.cos(th)
            y+=d*np.sin(th)
            plt.scatter(x, y, color="red",marker="x")
        elif node_id-1<N+W+R+A:
            x+=d*np.cos(2*th)
            y+=d*np.sin(2*th)
            plt.scatter(x, y, color="purple",marker="*")
        else:
            plt.scatter(x, y, color="blue",marker="s")
        # print(node,node_id)

        if node==node_id:
            # print("hi")
            plt.scatter(x, y, color="green",marker="p")
            plt.plot(x+1.5*np.cos(theta),y+1.5*np.sin(theta),color="green",linewidth=3)
    return
    
def highlight_node(matrices_and_info,requested_node):
    """
    given all of the matrices, plots the scenario on a white background
        Shows all nodes color coded
        Shows all connections color coded
        Shows river boundaries
        Highlights in green the requested_nodes

    Parameters
    ----------
    matrices_and_info :several
        outputs from the function get_matrices(filepath_name,width,length,nodes), 
        immediately unpacked below into adjacency matrix, distance matrix,
        river (edges), desc(riptions), node_loc(ations), width, length, and nodes
    requested_node : int
        node you wish to highlight

    """

    adjacency, distance, river, desc, node_loc, width, length, nodes = matrices_and_info
    

    plt.clf()
    fig, ax = plt.subplots()
    "iterate through adjacency matrix, ploting nodes and connections"
    [N,W,R,A,Riv,T]=desc[0]

    #Plot nodes
    d,th=get_offset(width, length)
    
    # for row in node_loc:
    #     [node_id,x,y,w,r,a]=row
    #     if node_id-1<N:
    #         plt.scatter(x,y,color="black",marker="o")
    #     elif node_id-1<N+W:
    #         plt.scatter(x+d, y, color="blue",marker="s")
    #     elif node_id-1<N+W+R:
    #         plt.scatter(x+d*np.cos(th), y+d*np.sin(th), color="red",marker="x")
    #     elif node_id-1<N+W+R+A:
    #         plt.scatter(x+d*np.cos(2*th), y+d*np.sin(2*th), color="purple",marker="*")
    #     else:
    #         plt.scatter(x+d, y, color="blue",marker="s")
    
    # plot connections
    color_dict={"II":"black","WW":'blue','RR':'red','AA':'purple','IA':"black"
                ,'AI':"black",'IW':"black",'WI':"black",'IR':"black",'RI':"black"}   
    for i in range(T):
        x1=node_loc[i][1]
        y1=node_loc[i][2]
        if i<N:
            start="I"
        elif i<N+W:
            x1+=d
            start="W"
        elif i<N+W+R:
            x1+=d*np.cos(th)
            y1+=d*np.sin(th)
            start="R"
        elif i<N+W+R+A:
            x1+=d*np.cos(2*th)
            y1+=d*np.sin(2*th)
            start="A"
        else:
            start="W"
        for j in range(i+1):
            # print(adjacency[i][j])
            if adjacency[i][j]==1:
                x2=node_loc[j][1]
                y2=node_loc[j][2]
                # print(x1,x2,y1,y2)
                if j<N:
                    end="I"
                elif j<N+W:
                    x2+=d
                    end="W"
                elif j<N+W+R:
                    x2+=d*np.cos(th)
                    y2+=d*np.sin(th)
                    end="R"
                elif j<N+W+R+A:
                    x2+=d*np.cos(2*th)
                    y2+=d*np.sin(2*th)
                    end="A"
                else:
                    end="W"
                plt.plot([x1,x2],[y1,y2],color=color_dict[start+end])
    "plot river"
    [xf,yf]=np.transpose(river)
    river_length=int(len(xf)/2)
    plt.fill(xf, yf, color='lightblue', alpha=1)
    plt.plot(xf[:river_length], yf[:river_length],color="blue")
    plt.plot(xf[river_length:], yf[river_length:],color="blue")
    update_plot_nodes(requested_node,node_loc, d,th, [N,W,R,A,Riv,T])
    return 


def plot_scenario(matrices_and_info,show_air):
    """
    given all of the matrices, plots the scenario on a white background
        Shows all nodes color coded
        Shows all connections color coded
        Shows river boundaries

    Parameters
    ----------
    matrices_and_info : several
        outputs from the function get_matrices(filepath_name,width,length,nodes), 
        immediately unpacked below into adjacency matrix, distance matrix,
        river (edges), desc(riptions), node_loc(ations), width, length, and nodes
    """
    adjacency, distance, river, desc, node_loc, width, length, nodes = matrices_and_info
    plt.clf()
    "iterate through adjacency matrix, ploting nodes and connections"
    [N,W,R,A,Riv,T]=desc[0]
    # print(desc[0],len(node_loc))
    #Plot nodes
    d,th=get_offset(width, length)
    for row in node_loc:
        [node_id,x,y,w,r,a]=row
        if node_id-1<N:
            plt.scatter(x,y,color="black",marker="o")
        elif node_id-1<N+W:
            plt.scatter(x+d, y, color="blue",marker="s")
        elif node_id-1<N+W+R:
            plt.scatter(x+d*np.cos(th), y+d*np.sin(th), color="red",marker="x")
        elif node_id-1<N+W+R+A:
            plt.scatter(x+d*np.cos(2*th), y+d*np.sin(2*th), color="purple",marker="*")
        else:
            plt.scatter(x, y, color="blue",marker="s")
    
    # plot connections
    color_dict={"II":"black","WW":'blue','RR':'red','AA':'purple','IA':"black"
                ,'AI':"black",'IW':"black",'WI':"black",'IR':"black",'RI':"black"}   
    for i in range(T):
        x1=node_loc[i][1]
        y1=node_loc[i][2]
        if i<N:
            start="I"
        elif i<N+W:
            x1+=d
            start="W"
        elif i<N+W+R:
            x1+=d*np.cos(th)
            y1+=d*np.sin(th)
            start="R"
        elif i<N+W+R+A:
            x1+=d*np.cos(2*th)
            y1+=d*np.sin(2*th)
            start="A"
        else:
            start="W"
        for j in range(i+1):
            # print(adjacency[i][j])
            if adjacency[i][j]==1:
                x2=node_loc[j][1]
                y2=node_loc[j][2]
                # print(x1,x2,y1,y2)
                if j<N:
                    end="I"
                elif j<N+W:
                    x2+=d
                    end="W"
                elif j<N+W+R:
                    x2+=d*np.cos(th)
                    y2+=d*np.sin(th)
                    end="R"
                elif j<N+W+R+A:
                    x2+=d*np.cos(2*th)
                    y2+=d*np.sin(2*th)
                    end="A"
                else:
                    end="W"
                if show_air:
                    plt.plot([x1,x2],[y1,y2],color=color_dict[start+end])
                else:
                    if start+end !="AA":
                        plt.plot([x1,x2],[y1,y2],color=color_dict[start+end])
    "plot river"
    [xf,yf]=np.transpose(river)
    river_length=int(len(xf)/2)
    plt.fill(xf, yf, color='lightblue', alpha=1)
    plt.plot(xf[:river_length], yf[:river_length],color="blue")
    plt.plot(xf[river_length:], yf[river_length:],color="blue")
    
    return 

def get_matrices(filepath_name,width,length,nodes):
    """
    Reads all relevent files at filepath_name and returns them as seperate 
    matrices

    Parameters
    ----------
    filepath_name : str
        file path to relevant folder.
    width : int
        how many miles wide is the the scenario (check file names in scenario folder).
    length : int
        how many miles long the is the scenario (check file names in scenario folder).
    nodes : int
        number of intersections in the scenario (check file names in scenario folder).

    Returns
    -------
    adjacency : list of lists
        adjacency matrix for node field.
    distance : list of lists
        distance between nodes that are connected in adjacency matrix.
    river : list of lists
        x and y coordinates that describe the river as a polygon.
    descriptions : list
        contains the the values of ["Intersections","Water_nodes","rail_nodes","air_nodes","river_nodes","total_nodes"].
    node_loc : list of lists
        each row: [node_id, x coordinate, y coorindate, water? rail? air?].
    width : int
        how many miles wide is the the scenario (check file names in scenario folder).
    length : int
        how many miles long the is the scenario (check file names in scenario folder).
    nodes : int
        number of intersections in the scenario (check file names in scenario folder).

    """
    # read in csvs
    extended_path=filepath_name+"scenario_W"+str(width)+"_L"+str(length)+"_N"+str(nodes)+"_"
    with open(extended_path+"adjacency_matrix.csv", 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        adjacency=[list(map(int,map(float,row))) for row in csv_reader]
    with open(extended_path+"distance_matrix.csv", 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        distance=[list(map(float,row)) for row in csv_reader]
    with open(extended_path+"node_location_type.csv", 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        next(csv_reader)
        node_loc=[list(map(float,row)) for row in csv_reader]
    with open(extended_path+"river.csv", 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        river=[list(map(float,row)) for row in csv_reader]
    with open(extended_path+"descriptions.csv", 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        next(csv_reader)
        descriptions = [list(map(int,map(float,row))) for row in csv_reader]
    return adjacency, distance, river, descriptions, node_loc, width, length, nodes



"""
---------------------------------------------------------------------------------------------------------------
For generating the node fields
---------------------------------------------------------------------------------------------------------------
"""
def generate_node_field_new(width,length, nodes, dice_roll=rnd.random(), curve=rnd.randint(1, 10), river_width=rnd.random()*8+6):
    locs=[]
    node_tracker=np.zeros((nodes+20,5))
    river=generate_river(width, length, dice_roll, curve, river_width)
    [As,Ts,ds,v1,r0,scale,xf,yf,x_mid,y_mid]=river
    river_nodes=[(x_mid[i*50],y_mid[i*50]) for i in range(20)]
    for i in range(20):
        node_tracker[nodes+i]=[river_nodes[i][0],river_nodes[i][1],1,0,0]
    poly=[(xf[i],yf[i]) for i in range(len(xf))] 
    numcities=rnd.randint(5,15)
    stdvs=[]
    city_center_x=[]
    city_center_y=[]
    for c in range(numcities):
        stdvs.append(15+30*rnd.random())
        city_center_x.append(width*rnd.random())
        city_center_y.append(length*rnd.random())
    water=[]
    rail=[]
    air=[]
    
    for i in range(nodes):
        which_city=rnd.random()
        for c in range(numcities):
            # print(numcities,(c+1)/numcities)
            if which_city<(c+1)/numcities:    
                
                x=np.random.normal(city_center_x[c],stdvs[c])
                while x>width or x<0:
                    x=np.random.normal(city_center_x[c],stdvs[c])
                y=np.random.normal(city_center_y[c],stdvs[c])
                while y>length or y<0:
                    y=np.random.normal(city_center_y[c],stdvs[c])
                
                if Polygon(poly).contains(Point(x,y)):
                    up_or_down=rnd.random()
                    node_tracker[i][2]=1
                    water.append(i)
                while Polygon(poly).contains(Point(x,y)):
                    if up_or_down>=.5:
                        y+=.1
                    else:
                        y-=.1
                node_tracker[i][0]=x
                node_tracker[i][1]=y
                
                dice_roll=rnd.random()
                if dice_roll<=node_type['rail']:
                   node_tracker[i][3]=1
                   rail.append(i)
                if dice_roll<=node_type['air']    :
                   node_tracker[i][-1]=1
                   air.append(i)
                break
            # print("City",c,city_center_x[c],city_center_y[c],node_tracker[i])
        locs.append(node_tracker[i][0:2])

    distances=[]
    for i in range(len(locs)):
        distances.append([])
        for loc in locs:
            if not np.array_equiv(locs[i], loc):
                distances[i].append(np.linalg.norm(np.array(loc)-np.array(locs[i])))
            else:
                distances[i].append(width*length)
    W=len(water)
    R=len(rail)       
    A=len(air)     
    Riv=len(river_nodes)
    T=nodes+W+R+A+Riv
    adjacency_matrix=np.zeros((T,T))         
    for node_id in range(nodes):
        num_neighbors=rnd.randint(2,5)
        k_nearest_neighbors=get_arg_kmin(distances[node_id],num_neighbors)
        # print(k_nearest_neighbors)
        for k in range(num_neighbors):
            adjacency_matrix[node_id][k_nearest_neighbors[k]]=1
            adjacency_matrix[k_nearest_neighbors[k]][node_id]=1
    k1=0
    k2=0
    k3=0
    for node_id in range(nodes):
        
        if node_tracker[node_id][2]==1:
            adjacency_matrix[node_id][nodes+k1]=1
            adjacency_matrix[nodes+k1][node_id]=1
            k1+=1
            
        if node_tracker[node_id][3]==1:
            adjacency_matrix[node_id][nodes+k2+W]=1
            adjacency_matrix[nodes+k2+W][node_id]=1
            k2+=1
        if node_tracker[node_id][4]==1:
            adjacency_matrix[node_id][nodes+k3+W+R]=1
            adjacency_matrix[nodes+k3+W+R][node_id]=1
            k3+=1

    dist_to_river=[]
    closest_river_node=[]
    for i in range(W):
        loc1=node_tracker[water[i]][:2]
        dist_to_river.append([])
        for j in range(20):
            loc2=river_nodes[j]
            dist_to_river[i].append(np.linalg.norm(loc1-loc2))
        closest_river_node.append(np.argmin(dist_to_river[i]))
        adjacency_matrix[nodes+i][T-Riv+closest_river_node[-1]]=1
        adjacency_matrix[T-Riv+closest_river_node[-1]][nodes+i]=1
                   
    
    for i in range(Riv-1):
        adjacency_matrix[T-Riv+i][T-Riv+i+1]=1
        adjacency_matrix[T-Riv+i+1][T-Riv+i]=1
        
    for a1 in range(A):
        for a2 in range(A):
            if a1!=a2:
                adjacency_matrix[nodes+W+R+a1][nodes+W+R+a2]=1
                adjacency_matrix[nodes+W+R+a2][nodes+W+R+a1]=1
    # for node_id in water:
    #     dist_from_start.append(np.linalg.norm(start-np.array(node_tracker[node_id][0:2])))
    # ordered_from_start=get_arg_kmin(dist_from_start, len(dist_from_start))
    # for i in range(len(ordered_from_start)-1):
    #     adjacency_matrix[water[ordered_from_start[i]]][ordered_from_start[i+1]]=1
    #     adjacency_matrix[ordered_from_start[i+1]][water[ordered_from_start[i]]]=1
    rail_dist=[]
    for i in range(R):
        rail_dist.append([])
        for j in range(R):
            if i!=j:
                rail_dist[i].append(distances[rail[i]][rail[j]])
            else:
                rail_dist[i].append(9000)
    # print(rail_dist)
    for node_id in range(R):
        num_neighbors=min(rnd.randint(1, 3),R)
        k_nearest_neighbors=get_arg_kmin(rail_dist[node_id],num_neighbors)
        
        for k in range(num_neighbors):
            # print(node_id,node_tracker[node_id][0:2],k,node_tracker[k_nearest_neighbors[k]][0:2])
            adjacency_matrix[nodes+W+node_id][nodes+W+k_nearest_neighbors[k]]=1
            adjacency_matrix[nodes+W+k_nearest_neighbors[k]][nodes+W+node_id]=1  
            
    # for i in range(len(adjacency_matrix)):
    #     line=""
    #     for j in range(len(adjacency_matrix)):
    #         line+=" "+str(adjacency_matrix[i][j])
    #     print(line)
    return node_tracker, distances, adjacency_matrix, river, [water,rail,air]

    







def plot_path(path, node_loc, width, length, path_name, path_num,number_of_paths):

    node_id,x,y,water,rail,air=node_loc[path[0]+1]
    xs=[x]
    ys=[y]
    types=[]
    if water>0:
        types.append(1)
    elif rail>0:
        types.append(2)
    elif air>0:
        types.append(3)
    else:
        types.append(0)
    d,th=get_offset(width,length) 
    colors=[(0,(i+1)/number_of_paths,0) for i in range(number_of_paths)]
    for node in path[1:]:
        node_id,x,y,water,rail,air=node_loc[node+1]
        xs.append(x)
        ys.append(y)
        if water>0:
            types.append(1)
            xs[-1]+=d
        elif rail>0:
            types.append(2)
            xs[-1]+=d*np.cos(th)
            ys[-1]+=d*np.sin(th)
        elif air>0:
            types.append(3)
            xs[-1]+=d*np.cos(2*th)
            ys[-1]+=d*np.sin(2*th)
        else:
            types.append(0)
    plt.plot(xs,ys,linestyle="dashed",color=colors[path_num],label=path_name)
    plt.legend()
    
    return 
    
def generate_scenarios(width,length,nodes,k):

    for i in range(k):
        path=r"Scenarios\scenario_"+str(i+1)
        if not os.path.exists(path):
            os.makedirs(path)
        name=path+r"\scenario_W"+str(width)+"_L"+str(length)+"_N"+str(nodes)
        print(name)
        get_outputs(width, length, nodes, name)
        
    return

def get_outputs(width,length,nodes,name="scenario"):
    node_field, distances, Adj_Matrix, river, [water, rail, air] = generate_map(width, length, nodes,True,name)
    N=nodes
    W=len(water)
    R=len(rail)
    A=len(air)
    Riv=20
    T=N+W+R+A+Riv
    nf=node_field[:nodes].tolist()
    for w_node in water:
        [x,y,w,r,a]=node_field[w_node]
        nf.append([x,y,1,0,0])
    for r_node in rail:
        [x,y,w,r,a]=node_field[r_node]
        nf.append([x,y,0,1,0])
    for a_node in air:
        [x,y,w,r,a]=node_field[a_node]
        nf.append([x,y,0,0,1])
    for riv_node in range(Riv):
        [x,y,w,r,a]=node_field[N+riv_node]
        nf.append([x,y,1,0,0])
    for node in range(N):
        [x,y,w,r,a]=node_field[node]
        nf[node]=[x,y,0,0,0]
    final_node_field=[]
    
    for node in range(T):
        [x,y,w,r,a]=nf[node]
        final_node_field.append([node+1,x,y,w,r,a])
    D_Matrix=np.zeros((T,T))

    """
    Fix distnaces matrix and this next looping code.
    """
    for i in range(T):

        for j in range(T):
            if Adj_Matrix[i][j]==1:
                loc1=np.array(final_node_field[i][:2])
                loc2=np.array(final_node_field[j][:2])
                D_Matrix[i][j]=np.linalg.norm(loc1-loc2)
                # if i<N:
                #     if j<N:
                #         D_Matrix[i][j]=distances[i][j]
                #     elif j<N+W:
                #         D_Matrix[i][j]=distances[i][water[N+W-j-1]]
                #     elif j<N+W+R:
                #         D_Matrix[i][j]=distances[i][rail[N+W+R-j-1]]
                #     elif j<N+W+R+A:
                #         D_Matrix[i][j]=distances[i][air[N+W+R+A-j-1]]
                #     else:
                #         loc1=np.array(final_node_field[i][:2])
                #         loc2=np.array(final_node_field[j][:2])
                #         D_Matrix[i][j]=np.linalg.norm(loc1-loc2)
                # elif i<N+W:
                #     if j<N:
                #         D_Matrix[i][j]=distances[water[N+W-i-1]][j]
                #     elif j<N+W:
                #         D_Matrix[i][j]=distances[water[N+W-i-1]][water[N+W-j-1]]
                #     elif j<N+W+R:
                #         D_Matrix[i][j]=distances[water[N+W-i-1]][rail[N+W+R-j-1]]
                #     elif j<N+W+R+A:
                #         D_Matrix[i][j]=distances[water[N+W-i-1]][air[N+W+R+A-j-1]]
                #     else:
                #         D_Matrix[i][j]=distances[water[N+W-i-1]][j]
                # elif i<N+W+R:
                #     if j<N:
                #         # print(i,N+W+R-i-1,rail)
                #         D_Matrix[i][j]=distances[rail[N+W+R-i-1]][j]
                #     elif j<N+W:
                #         D_Matrix[i][j]=distances[rail[N+W+R-i-1]][water[N+W-j-1]]
                #     elif j<N+W+R:
                #         D_Matrix[i][j]=distances[rail[N+W+R-i-1]][rail[N+W+R-j-1]]
                #     else:
                #         D_Matrix[i][j]=distances[rail[N+W+R-i-1]][air[N+W+R+A-j-1]]
                # else:
                #     if j<N:
                #         D_Matrix[i][j]=distances[air[N+W+R+A-i-1]][j]
                #     elif j<N+W:
                #         D_Matrix[i][j]=distances[air[N+W+R+A-i-1]][water[N+W-j-1]]
                #     elif j<N+W+R:
                #         D_Matrix[i][j]=distances[air[N+W+R+A-i-1]][rail[N+W+R-j-1]]
                #     else:
                #         D_Matrix[i][j]=distances[air[N+W+R+A-i-1]][air[N+W+R+A-j-1]]
    [As, Ts, ds, v1, r0, scale, xf, yf, x_mid,y_mid]=river
    river_nodes=[(x_mid[i*50],y_mid[i*50]) for i in range(20)]
    with open(name+'_node_location_type.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Node_ID","x","y","Water","Rail","Air"])
        writer.writerows(final_node_field)
    with open(name+'_adjacency_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(Adj_Matrix)
    with open(name+'_distance_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(D_Matrix)
    with open(name+'_descriptions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Intersections","Water_nodes","rail_nodes","air_nodes","river_nodes","total_nodes"])
        writer.writerow([N,W,R,A,Riv,T])
    with open(name+'_river.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(xf)):  
            writer.writerow([xf[i],yf[i]])
    with open(name+'_river_nodes.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(river_nodes)

    return 
    
def generate_map(width,length,nodes,save=False,name=''):
    plt.clf()
    
    plt.gca().set_aspect('equal')
    
    river_width=rnd.random()*8+6
    curve=rnd.randint(2, 10)
    
    node_field,distances,Adj,river,special_nodes=generate_node_field_new(width,length,nodes, rnd.random(), curve, river_width)
    [As, Ts, ds, v1, r0, scale, xf, yf, x_mid,y_mid]=river
    river_nodes=[(x_mid[i*50],y_mid[i*50]) for i in range(20)]
    poly=[(xf[i],yf[i]) for i in range(len(xf))] 
    plt.xlim((0,width))
    plt.ylim((0,length))
    d,th=get_offset(width,length)
    colors=["blue","red","purple"]
    markers=["s","x","*"]
    node_id=0

    rail=[]
    for node in node_field:
        # print(node)
        x=node[0]
        y=node[1]
        # print(x,y)
        plt.scatter(x,y,color='black',marker="o")
        
        

        if node[2]==1:
            i=0
            # print("hi")
            if not Polygon(poly).contains(Point(x,y)):
                xs=np.linspace(x, x+d*np.cos(th*i),2)
                ys=np.linspace(y, y+d*np.sin(th*i),2)
                plt.plot(xs,ys,color="black",linestyle='dotted')
                x=x+d*np.cos(th*i)
                y=y+d*np.sin(th*i)
            plt.scatter(x,y,color=colors[i],marker=markers[i])
            
        if node[3]==1:
            i=1
            locx=x+d*np.cos(th*i)
            locy=y+d*np.sin(th*i)
            rail.append([locx,locy])
            plt.scatter(x+d*np.cos(th*i),y+d*np.sin(th*i),color=colors[i],marker=markers[i])
            xs=np.linspace(x, x+d*np.cos(th*i),2)
            ys=np.linspace(y, y+d*np.sin(th*i),2)
            plt.plot(xs,ys,color="black",linestyle='dotted')
        if node[4]==1:
            i=2
            plt.scatter(x+d*np.cos(th*i),y+d*np.sin(th*i),color=colors[i],marker=markers[i])
            xs=np.linspace(x, x+d*np.cos(th*i),2)
            ys=np.linspace(y, y+d*np.sin(th*i),2)
            plt.plot(xs,ys,color="black",linestyle='dotted')
        node_id+=1

    for i in range(nodes):
        loc=node_field[i][0:2]
        for j in range(nodes):
            if Adj[i][j]==1:
                loc2=node_field[j][0:2]
                xs=np.linspace(loc[0],loc2[0],2)
                ys=np.linspace(loc[1],loc2[1],2)
                plt.plot(xs,ys,color="black")
    

    W=len(special_nodes[0])
    R=len(special_nodes[1])
    for i in range(R):
        loc=np.array(node_field[special_nodes[1][i]][0:2])+np.array([d*np.cos(th),d*np.sin(th)])
        for j in range(R):
            # print(A[node][nodes+len(special_nodes[0]):nodes+len(special_nodes[0])+len(special_nodes[1])])
            if Adj[nodes+W+i][nodes+W+j]==1:
                
                loc2=np.array(node_field[special_nodes[1][j]][0:2])+np.array([d*np.cos(th),d*np.sin(th)])
                # print("hi",loc,loc2)
                xs=np.linspace(loc[0],loc2[0],2)
                ys=np.linspace(loc[1],loc2[1],2)
                plt.plot(xs,ys,color="red")
    T=len(Adj)
    Riv=len(river_nodes)
    # print(T,Riv,len(node_field))
    for i in range(Riv):
        for j in range(W):
            if Adj[T-Riv+i][nodes+j]==1:
                [x1,y1]=river_nodes[i]
                [x2,y2]=np.array(node_field[special_nodes[0][j]][:2])+np.array([d,0])
                plt.plot([x1,x2],[y1,y2],color="orange")
        for k in range(Riv):
            if Adj[T-Riv+i][T-Riv+k]==1:
                [x1,y1]=river_nodes[i]
                [x2,y2]=river_nodes[k]
                plt.plot([x1,x2],[y1,y2],color="blue")

    if save:
        plt.tight_layout()
        plt.savefig(name+"_figure.png")
    return node_field, distances, Adj, river, special_nodes

def get_arg_kmin(vec,k):
    M=10000
    args=[]
    while len(args)<k:
        args.append(np.argmin(vec))
        vec[args[-1]]=M
    return args
        
def get_offset(width,length):
    d=min(width,length)/100
    th=2*np.pi/(L-1)
    return d, th



def generate_river(width,length,dice_roll,curve,river_width):

    plt.clf()
    ax=plt.axes()
    ax.set_facecolor("green")
    if dice_roll>.5:
        x1t=rnd.random()*width
        x2t=rnd.random()*width
        x2=max(x1t,x2t)
        x1=min(x1t,x2t)
        y1=0
        y2=length
    else:
       y1=rnd.random()*length
       y2=rnd.random()*length
       x1=0
       x2=width
    As=[]
    Ts=[]
    ds=[]
    for i in range(curve):
        As.append(2+rnd.random()*10)
        Ts.append(.4+rnd.random()*.6)
        ds.append(rnd.random()*2) 
    t=np.linspace(0,1,1000)
    v1=np.array([x2-x1,y2-y1])
    r0=np.array([x1,y1])
    vx=v1[0]*t+r0[0]
    vy=v1[1]*t+r0[1]

    v2=rot_matrix(np.pi/2)@v1
    scale=v2/np.linalg.norm(v2)
    for i in range(curve):
        vx+=scale[0]*(As[i]*np.sin((np.pi*2/Ts[i])*t)+ds[i])
        vy+=scale[1]*(As[i]*np.sin((np.pi*2/Ts[i])*t)+ds[i])

    vx_shift=vx+river_width*scale[0]
    vy_shift=vy+river_width*scale[1]
    x_mid=vx+river_width*scale[0]*.5
    y_mid=vy+river_width*scale[1]*.5
    xf = np.concatenate((vx, vx_shift[::-1]) )
    yf= np.concatenate((vy, vy_shift[::-1]) )
    
   
    plt.fill(xf, yf, color='lightblue', alpha=1)
    plt.plot(vx, vy,color="blue")
    plt.plot(vx+river_width*scale[0],vy+river_width*scale[1],color="blue")
    # plt.fill_between(vx, vy, vy+river_width*scale[1],color="blue")
    return [As, Ts, ds, v1, r0, scale, xf, yf, x_mid, y_mid]

def rot_matrix(theta):
    R= np.array([[np.cos(theta),-np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    return R


"""
Old Functions
"""
def generate_node_field(width,length, nodes, dice_roll=rnd.random(), curve=rnd.randint(1, 10), river_width=rnd.random()*8+6):
    locs=[]
    node_tracker=np.zeros((nodes,5))
    river=generate_river(width, length, dice_roll, curve, river_width)
    [As,Ts,ds,v1,r0,scale,xf,yf,x_mid,y_mid]=river
    river_nodes=[(x_mid[i*50],y_mid[i*50]) for i in range(20)]
    for i in range(20):
        np.append(node_tracker,[nodes+i+1,river_nodes[i][0],river_nodes[i][1],1,0,0])
    poly=[(xf[i],yf[i]) for i in range(len(xf))] 
    numcities=rnd.randint(5,15)
    stdvs=[]
    city_center_x=[]
    city_center_y=[]
    for c in range(numcities):
        stdvs.append(15+30*rnd.random())
        city_center_x.append(width*rnd.random())
        city_center_y.append(length*rnd.random())
    water=[]
    rail=[]
    air=[]
    
    for i in range(nodes):
        which_city=rnd.random()
        for c in range(numcities):
            # print(numcities,(c+1)/numcities)
            if which_city<(c+1)/numcities:    
                
                x=np.random.normal(city_center_x[c],stdvs[c])
                while x>width or x<0:
                    x=np.random.normal(city_center_x[c],stdvs[c])
                y=np.random.normal(city_center_y[c],stdvs[c])
                while y>length or y<0:
                    y=np.random.normal(city_center_y[c],stdvs[c])
                
                if Polygon(poly).contains(Point(x,y)):
                    up_or_down=rnd.random()
                    node_tracker[i][2]=1
                    water.append(i)
                while Polygon(poly).contains(Point(x,y)):
                    if up_or_down>=.5:
                        y+=.1
                    else:
                        y-=.1
                node_tracker[i][0]=x
                node_tracker[i][1]=y
                
                dice_roll=rnd.random()
                if dice_roll<=node_type['rail']:
                   node_tracker[i][3]=1
                   rail.append(i)
                if dice_roll<=node_type['air']    :
                   node_tracker[i][-1]=1
                   air.append(i)
                break
            # print("City",c,city_center_x[c],city_center_y[c],node_tracker[i])
        locs.append(node_tracker[i][0:2])

    distances=[]
    for i in range(len(locs)):
        distances.append([])
        for loc in locs:
            if not np.array_equiv(locs[i], loc):
                distances[i].append(np.linalg.norm(np.array(loc)-np.array(locs[i])))
            else:
                distances[i].append(width*length)
    W=len(water)
    R=len(rail)       
    A=len(air)     
    adjacency_matrix=np.zeros((nodes+W+R+len(air),nodes+W+R+len(air)))         
    for node_id in range(nodes):
        num_neighbors=rnd.randint(2,5)
        k_nearest_neighbors=get_arg_kmin(distances[node_id],num_neighbors)
        # print(k_nearest_neighbors)
        for k in range(num_neighbors):
            adjacency_matrix[node_id][k_nearest_neighbors[k]]=1
            adjacency_matrix[k_nearest_neighbors[k]][node_id]=1
    k1=0
    k2=0
    k3=0
    for node_id in range(nodes):
        
        if node_tracker[node_id][2]==1:
            adjacency_matrix[node_id][nodes+k1]=1
            adjacency_matrix[nodes+k1][node_id]=1
            k1+=1
            
        if node_tracker[node_id][3]==1:
            adjacency_matrix[node_id][nodes+k2+W]=1
            adjacency_matrix[nodes+k2+W][node_id]=1
            k2+=1
        if node_tracker[node_id][4]==1:
            adjacency_matrix[node_id][nodes+k3+W+R]=1
            adjacency_matrix[nodes+k3+W+R][node_id]=1
            k3+=1

    dist_to_river=[]
    closest_river_node=[]
    for i in range(W):
        loc1=node_tracker[water[i]][:2]
        dist_to_river.append([])
        for j in range(20):
            loc2=river_nodes[j]
            dist_to_river[i].append(np.linalg.norm(loc1-loc2))
        closest_river_node.append(np.argmin(dist_to_river[i]))
    # print(closest_river_node,W)
    water_dist=[]
    for i in range(W):
        water_dist.append([])
        for j in range(W):
            if i!=j:
                water_dist[i].append(distances[water[i]][water[j]])
            else:
                water_dist[i].append(99999)
    for node_id in range(W):
        num_neighbors=min(rnd.randint(1, 3),W)
        k_nearest_neighbors=get_arg_kmin(water_dist[node_id],num_neighbors)
        for k in range(num_neighbors):
            adjacency_matrix[nodes+node_id][nodes+k_nearest_neighbors[k]]=1
            adjacency_matrix[nodes+k_nearest_neighbors[k]][nodes+node_id]=1              
    
    for a1 in range(A):
        for a2 in range(A):
            if a1!=a2:
                adjacency_matrix[nodes+W+R+a1][nodes+W+R+a2]=1
                adjacency_matrix[nodes+W+R+a2][nodes+W+R+a1]=1
    # for node_id in water:
    #     dist_from_start.append(np.linalg.norm(start-np.array(node_tracker[node_id][0:2])))
    # ordered_from_start=get_arg_kmin(dist_from_start, len(dist_from_start))
    # for i in range(len(ordered_from_start)-1):
    #     adjacency_matrix[water[ordered_from_start[i]]][ordered_from_start[i+1]]=1
    #     adjacency_matrix[ordered_from_start[i+1]][water[ordered_from_start[i]]]=1
    rail_dist=[]
    for i in range(R):
        rail_dist.append([])
        for j in range(R):
            if i!=j:
                rail_dist[i].append(distances[rail[i]][rail[j]])
            else:
                rail_dist[i].append(9000)
    # print(rail_dist)
    for node_id in range(R):
        num_neighbors=min(rnd.randint(1, 3),R)
        k_nearest_neighbors=get_arg_kmin(rail_dist[node_id],num_neighbors)
        
        for k in range(num_neighbors):
            # print(node_id,node_tracker[node_id][0:2],k,node_tracker[k_nearest_neighbors[k]][0:2])
            adjacency_matrix[nodes+W+node_id][nodes+W+k_nearest_neighbors[k]]=1
            adjacency_matrix[nodes+W+k_nearest_neighbors[k]][nodes+W+node_id]=1  
            
    # for i in range(len(adjacency_matrix)):
    #     line=""
    #     for j in range(len(adjacency_matrix)):
    #         line+=" "+str(adjacency_matrix[i][j])
    #     print(line)
    return node_tracker, distances, adjacency_matrix, river, [water,rail,air]