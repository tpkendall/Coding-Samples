# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 21:18:45 2023

@author: thomas.kendall
"""
import numpy as np
import random as random
import matplotlib.pyplot as plt

w=10
l=10
s=10
viable_nodes = [i+j*w+1 for i in range(w-1) for j in range(w-1)]

def get_points(wide=w, long=l, delta=s):
    """
    Generates node field

    Parameters
    ----------
    wide : Int, optional
        number of nodes wide in the node field. The default is w.
    long : Int, optional
        number of nodes long in the node field.. The default is l.
    delta : Float, optional
        step size between nodes. The default is s.

    Returns
    -------
    points : dictionary
        node dictionary {node id : (x, y, z), ...}.

    """
    points= {i+j*wide+1 : [i*delta, j*delta, 10*random.random()] for i in range(wide) for j in range(long)}
    return points
    
def get_node_id(x, y, points, wide=w, long=l, delta=s):
    """
    Gets the node id that is the bottom left corner of the node square
    containing the point (x,y)

    Parameters
    ----------
    x : float
        x coordinate.
    y : float
        y coordinate.
    points : dictionary
        node dictionary {node id : (x, y, z), ...}.
    wide : Int, optional
        number of nodes wide in the node field. The default is w.
    long : Int, optional
        number of nodes long in the node field.. The default is l.
    delta : Float, optional
        step size between nodes. The default is s.

    Returns
    -------
    node : Int
        bottom left corner node of the node square
        containing the point (x,y).
    points[node] : list
        coordinate of bottom left corner node.

    """
    for node in points:
        if node in viable_nodes:
            [xn,yn,en] = points[node]
            [xnc,ync,enc] = points[node+wide+1]
            if x>=xn and y>=yn and x<xnc and y<ync:
                return node, points[node]
    print((x,y), "oh shit")
    return


def plot_spline(intensity=1, wide=w, long=l, delta=s):
    """
    Demonstrates the 3D smoothing

    Parameters
    ----------
    intensity : Int, optional
        The higher the intensity, the more points the graph samples at. The default is 1.
    wide : Int, optional
        number of nodes wide in the node field. The default is w.
    long : Int, optional
        number of nodes long in the node field.. The default is l.
    delta : Float, optional
        step size between nodes. The default is s.

    Returns
    -------
    None.

    """
    points = get_points(wide, long, delta)
    xs=np.linspace(0, delta*(wide-2), intensity*wide)
    ys=np.linspace(0, delta*(long-2), intensity*long)
    X, Y = np.meshgrid(xs, ys)
    # Z = get_val(np.ravel(X), np.ravel(Y), points)
    Z=[]
    for y in ys:
        Z.append([])
        for x in xs:
            Z[-1].append(get_val(x, y, points))
        
    plt.clf()
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, np.array(Z))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return

def get_val(x, y, points, wide=w, long=l, delta=s):
    """
    Gets interpolated z value at coordinate (x, y)

    Parameters
    ----------
    x : float
        x coordinate.
    y : float
        y coordinate.
    points : dictionary
        node dictionary {node id : (x, y, z), ...}.
    wide : Int, optional
        number of nodes wide in the node field. The default is w.
    long : Int, optional
        number of nodes long in the node field.. The default is l.
    delta : Float, optional
        step size between nodes. The default is s.

    Returns
    -------
    val : Float
        z coordinate interpolated using the node square containing (x,y).

    """
    n, [x1, y1, e1] = get_node_id(x, y, points)
    
    [x2, y1, e2] = points[n+1]
    [x1, y2, e3] = points[n+wide]
    [x2, y2, e4] = points[n+wide+1]
    val = e1 + (e3 - e1) * (y - y1) / (y2 - y1) + (e2 - e1) * (x - x1) / (x2 - x1) + ((e4 - e3) - (e2 - e1)) * ((x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))
    return val
    
def check_nodes(xval, yval, wide=w, long=l, delta=s):
    """
    shows the node field and the node square containing (xval, yval)

    Parameters
    ----------
    xval : float
        x coordinate.
    yval : float
        y coordinate.
    wide : Int, optional
        number of nodes wide in the node field. The default is w.
    long : Int, optional
        number of nodes long in the node field.. The default is l.
    delta : Float, optional
        step size between nodes. The default is s.

    Returns
    -------
    None.

    """
    if xval>delta*(wide-1) or yval > delta*(long-1):
        print("No, you cant do that")
        return
    points = get_points(wide, long, delta)
    [x,y,e] = np.transpose([points[i+1] for i in range(wide*long)])
    plt.clf()
    plt.scatter(x,y)
    N, [xn, yn, en] = get_node_id(xval, yval, points)
    [xc,yc,ec] = points[N+wide+1]
    plt.scatter(xval, yval, marker="*")
    plt.scatter(xn, yn, color='blue')
    plt.scatter(xc, yc, color='green')
    plt.plot([xn, xc], [yn, yn], color='black')
    plt.plot([xn, xc], [yc, yc], color='black')
    plt.plot([xn, xn], [yn, yc], color='black')
    plt.plot([xc, xc], [yn, yc], color='black')
    return

def check_nodes2(N, wide=w, long=l, delta=s):
    """
    shows the node field and the node square characterized by the bottom left
    corner node being N

    Parameters
    ----------
    N : Int
        bottom left corner node of node square.
    wide : Int, optional
        number of nodes wide in the node field. The default is w.
    long : Int, optional
        number of nodes long in the node field.. The default is l.
    delta : Float, optional
        step size between nodes. The default is s.

    Returns
    -------
    None.

    """
    if N not in viable_nodes:
        print("No, you cant do that")
        return
    points = get_points(wide, long, delta)
    [x,y,e] = np.transpose([points[i+1] for i in range(wide*long)])
    plt.clf()
    plt.scatter(x,y)
    [xn,yn,en]= points[N]
    [xc,yc,ec] = points[N+wide+1]
    plt.scatter(xn, yn, color='blue')
    plt.scatter(xc, yc, color='green')
    plt.plot([xn, xc], [yn, yn], color='black')
    plt.plot([xn, xc], [yc, yc], color='black')
    plt.plot([xn, xn], [yn, yc], color='black')
    plt.plot([xc, xc], [yn, yc], color='black')
    plt.show()
    return