# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:32:55 2023

@author: aidan.looney
"""

import numpy as np
import matplotlib.pyplot as plt
# from mycolorpy import colorlist as mcp
import matplotlib.animation as animation

l1=.3698873002608578;l2=.2143123842713125;v0=905.256;g=9.807;hr = 0.06604;

p = 1.225; d = 0.00556; c = .27; m = 62/15432.4;

A = np.pi*((d**2)/4)
k = c*p*A/(2*m)
"""
Notes:
    o All angles are in radians
    o Possible "error_methods"
        -TEM: Total Error Method (all error is positve)
        -SEM: Squared Error Method (square each error)
        -NEM: Total Error Method (collect all error allowing for negatives (the true integral))
    o Try go_fishing(173*np.pi/180,133*np.pi/180,300,True,"NEM")
"""
def compare_error_methods(alpha,beta,r):
    plt.clf()
    error_methods=["NEM","TEM","SEM"]
    plt.clf()
    fig1, axs = plt.subplots(2,1, sharex='col')
    fig1.suptitle("\u03B1 ="+str(np.round(180*alpha/np.pi,decimals=2))+"\u00b0, \u03B2="+str(np.round(180*beta/np.pi,decimals=2))+"\u00b0")
    for i in range(3):
        yfzf, err= go_fishing(alpha,beta,r,False,error_methods[i])
        x, y, z, vx_values, vy_values, vz_values, los, SE=runge_kutta_4th_order(yfzf[0], yfzf[1], alpha, beta, r,error_methods[i])
        axs[0].plot(x,y,label=error_methods[i])
        axs[1].plot(x,np.array(z)-np.array(los))
        print(error_methods[i],"complete")
    for i in range(2):
        axs[i].grid()
        axs[i].plot(x,0*np.array(x),color='black')
        axs[0].set_title('Lateral Displacement')
        axs[1].set_title('Vertical Displacement')
    axs[0].legend()
    return

def second_vis_fishing(alpha,beta,r,error_method):
    plt.clf()
    fig, ax = plt.subplots()
    center=[0,.065]
    xs=[center[0]]
    ys=[center[1]]
    d=.001
    accepteable_error=10**-8
    net=cast_net(center,d)
    errors,i,j,min_error=get_errors(net,alpha,beta,r,error_method)
    error=[min_error]
    iters=[1]
    
    while objective(error_method,error[-1],accepteable_error,d):
        d=.6*d
        center=net[i][j]
        xs.append(center[0])
        ys.append(center[1])
        net=cast_net(center,d)
        errors,i,j,min_error=get_errors(net,alpha,beta,r,error_method)
        error.append(min_error)
        iters.append(iters[-1]+1)
    ax.set_xlim([min(xs),max(xs)])
    ax.set_ylim([min(ys),max(ys)])
    ax.grid()
    ax.set_xlabel("yf")
    ax.set_ylabel("zf")
    scat=ax.scatter(xs[0],ys[0])
    
    def animate(i):
        ax.set_title("Iteration: "+str(i)+", Error="+str(np.format_float_scientific(error[i],unique=False,precision=4)))
        scat.set_offsets((xs[i],ys[i]))
        return scat
    ani = animation.FuncAnimation(fig, animate, repeat=False, frames=len(xs)-1)
    writer = animation.PillowWriter(fps=2,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
    ani.save('scatter.gif', writer=writer)
    plt.show()
    return


def go_fishing(alpha,beta,r=300,plot = None,error_method="NEM"):
    center=[0,.065]
    d=.001
    accepteable_error=10**-8
    net=cast_net(center,d)
    errors,i,j,min_error=get_errors(net,alpha,beta,r,error_method)
    error=[min_error]
    
    while objective(error_method,abs(error[-1]),accepteable_error,d):
        d=.6*d
        center=net[i][j]
        net=cast_net(center,d)
        errors,i,j,min_error=get_errors(net,alpha,beta,r,error_method)
        error.append(min_error)
    
    if plot:
        shot_plot(center[0],center[1],alpha,beta,r,error_method)
    return center, error[-1]

def objective(error_method,error,accepteable_error,d):
    if error_method == "SEM" or error_method == "TEM":
        return d>10**-10
    else:
        return error>accepteable_error

def cast_net(center,d):
    cx,cy=center
    net=[]
    dy=d
    for i in range(3):
        net.append([[cx-d,cy+dy],[cx,cy+dy],[cx+d,cy+dy]])
        dy-=d
    return net

def vis_go_fishing(alpha,beta,r,error_method):
    center=[0,.065]
    xs=[center[0]]
    ys=[center[1]]
    d=.001
    accepteable_error=10**-8
    net=cast_net(center,d)
    errors,i,j,min_error=get_errors(net,alpha,beta,r,error_method)
    error=[min_error]
    iters=[1]
    
    while objective(error_method,error[-1],accepteable_error,d):
        d=.6*d
        center=net[i][j]
        xs.append(center[0])
        ys.append(center[1])
        net=cast_net(center,d)
        errors,i,j,min_error=get_errors(net,alpha,beta,r,error_method)
        error.append(min_error)
        iters.append(iters[-1]+1)
    
    plt.clf()
    # color1=mcp.gen_color(cmap="RdYlGn",n=len(error))
    fig, axs = plt.subplots(1,2)
    axs[0].set_xlim(min(xs), max(xs))
    axs[0].set_ylim(min(ys), max(ys))
    axs[1].set_xlim(0, len(xs))
    axs[0].grid()
    axs[1].grid()
    axs[1].set_ylim(min(error), max(error))
    axs[1].set_yscale('log')
    # axs[0].scatter(xs[0],ys[0],c=color1[0])
    # axs[1].scatter(iters[0],error[0],c=color1[0])
    # for i in range(len(error)):
    #     axs[0].scatter(xs[i],ys[i],c=color1[i])
    #     axs[1].scatter(iters[i],error[i],c=color1[i])
    #     plt.draw()
    #     plt.pause(.1)
    axs[0].scatter(xs[0],ys[0],c="blue")
    axs[1].scatter(iters[0],error[0],c="blue")
    for i in range(len(error)):
        axs[0].scatter(xs[i],ys[i],c="blue")
        axs[1].scatter(iters[i],error[i],c="blue")
        plt.draw()
        plt.pause(.1)
        
        
    return center, error[-1]

    
    
def get_errors(net,alpha,beta,r,error_method):
    min_error=99999
    errors=[]
    for i in range(3):
        errors.append([])
        for j in range(3):
            yf,zf=net[i][j]
            errors[i].append(runge_kutta_4th_order(yf, zf, alpha, beta, r,error_method)[-1])
            if errors[i][-1]<min_error:
                min_error=errors[i][-1]
                istar=i
                jstar=j
    return errors, istar, jstar, min_error
                
                
def shot_plot(yf,zf,alpha,beta,r,error_method):
    x, y, z, vx_values, vy_values, vz_values, los, SE=runge_kutta_4th_order(yf, zf, alpha, beta, r,error_method)
    plt.clf()
    fig1, axs = plt.subplots(2,1, sharex='col')
    fig1.suptitle("Squared Error: "+str(SE))
    axs[0].plot(x,y)
    axs[1].plot(x,np.array(z)-np.array(los))
    axs[0].set_title('Lateral Displacement')
    axs[1].set_title('Vertical Displacement')
    for i in range(2):
        axs[i].grid()
        axs[i].plot(x,0*np.array(x),color='black')
    return


def runge_kutta_4th_order(yf, zf, alpha, beta, r = 300, error_method = "NEM"):
    dt=.001
    y0,z0=initial_position(yf, zf, alpha, beta)
    v0_vec=initial_velocity(yf, zf, alpha, beta)
    x_values = [0]
    y_values = [y0]
    z_values = [z0]
    vx_values = [v0_vec[0]]
    vy_values = [v0_vec[1]]
    vz_values = [v0_vec[2]]
    los = [0]
    r=r*np.cos(alpha)
    yerror=0
    zerror=0
    while abs(x_values[-1])<abs(r):
        x = x_values[-1]
        y = y_values[-1]
        z = z_values[-1]
        vx = vx_values[-1]
        vy = vy_values[-1]
        vz = vz_values[-1]

        ax1, ay1, az1 = acceleration(np.array([vx,vy,vz]))
        k1x = dt * vx
        k1y = dt * vy
        k1z = dt * vz
        k1vx = dt * ax1
        k1vy = dt * ay1
        k1vz = dt * az1

        ax2, ay2, az2 = acceleration(np.array([ vx + 0.5 * k1vx, vy + 0.5 * k1vy, vz + 0.5 * k1vz]))
        k2x = dt * (vx + 0.5 * k1vx)
        k2y = dt * (vy + 0.5 * k1vy)
        k2z = dt * (vz + 0.5 * k1vz)
        k2vx = dt * ax2
        k2vy = dt * ay2
        k2vz = dt * az2

        ax3, ay3, az3 = acceleration(np.array( [vx + 0.5 * k2vx, vy + 0.5 * k2vy, vz + 0.5 * k2vz]))
        k3x = dt * (vx + 0.5 * k2vx)
        k3y = dt * (vy + 0.5 * k2vy)
        k3z = dt * (vz + 0.5 * k2vz)
        k3vx = dt * ax3
        k3vy = dt * ay3
        k3vz = dt * az3

        ax4, ay4, az4 = acceleration(np.array( [vx + k3vx, vy + k3vy, vz + k3vz]))
        k4x = dt * (vx + k3vx)
        k4y = dt * (vy + k3vy)
        k4z = dt * (vz + k3vz)
        k4vx = dt * ax4
        k4vy = dt * ay4
        k4vz = dt * az4

        x_new = x + (1 / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y_new = y + (1 / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
        z_new = z + (1 / 6) * (k1z + 2 * k2z + 2 * k3z + k4z)
        vx_new = vx + (1 / 6) * (k1vx + 2 * k2vx + 2 * k3vx + k4vx)
        vy_new = vy + (1 / 6) * (k1vy + 2 * k2vy + 2 * k3vy + k4vy)
        vz_new = vz + (1 / 6) * (k1vz + 2 * k2vz + 2 * k3vz + k4vz)

        x_values.append(x_new)
        y_values.append(y_new)
        z_values.append(z_new)
        vx_values.append(vx_new)
        vy_values.append(vy_new)
        vz_values.append(vz_new)
        los.append(np.tan(alpha)*x_values[-1])
        if error_method == "SEM":
            yerror+=(y_values[-1]*abs(x_values[-2]-x_values[-1]))**2
            zerror+=((z_values[-1]-los[-1])*abs(x_values[-2]-x_values[-1]))**2
        elif error_method =="TEM":
            yerror+=abs(y_values[-1]*abs(x_values[-2]-x_values[-1]))
            zerror+=abs((z_values[-1]-los[-1])*abs(x_values[-2]-x_values[-1]))
        else:
            yerror+=(y_values[-1]*abs(x_values[-2]-x_values[-1]))
            zerror+=((z_values[-1]-los[-1])*abs(x_values[-2]-x_values[-1]))
            
    squared_error=yerror**2+zerror**2

    return x_values, y_values, z_values, vx_values, vy_values, vz_values, los, squared_error


def mess_with_runge_kutta_4th_order(yf, zf, alpha, beta, r = 300, error_method = "NEM"):
    dt=.001
    y0,z0=initial_position(yf, zf, alpha, beta)
    v0_vec=initial_velocity(yf, zf, alpha, beta)
    x_values = [0]
    y_values = [y0]
    z_values = [z0]
    vx_values = [v0_vec[0]]
    vy_values = [v0_vec[1]]
    vz_values = [v0_vec[2]]
    los = [0]
    r=r*np.cos(alpha)
    yerror=0
    zerror=0
    while abs(x_values[-1])<abs(r):
        x = x_values[-1]
        y = y_values[-1]
        z = z_values[-1]
        vx = vx_values[-1]
        vy = vy_values[-1]
        vz = vz_values[-1]

        ax1, ay1, az1 = acceleration(np.array([vx,vy,vz]))
        k1x = dt * vx
        k1y = dt * vy
        k1z = dt * vz
        k1vx = dt * ax1
        k1vy = dt * ay1
        k1vz = dt * az1

        ax2, ay2, az2 = acceleration(np.array([ vx + 0.5 * k1vx, vy + 0.5 * k1vy, vz + 0.5 * k1vz]))
        k2x = dt * (vx + 0.5 * k1vx)
        k2y = dt * (vy + 0.5 * k1vy)
        k2z = dt * (vz + 0.5 * k1vz)
        k2vx = dt * ax2
        k2vy = dt * ay2
        k2vz = dt * az2

        ax3, ay3, az3 = acceleration(np.array( [vx + 0.5 * k2vx, vy + 0.5 * k2vy, vz + 0.5 * k2vz]))
        k3x = dt * (vx + 0.5 * k2vx)
        k3y = dt * (vy + 0.5 * k2vy)
        k3z = dt * (vz + 0.5 * k2vz)
        k3vx = dt * ax3
        k3vy = dt * ay3
        k3vz = dt * az3

        ax4, ay4, az4 = acceleration(np.array( [vx + k3vx, vy + k3vy, vz + k3vz]))
        k4x = dt * (vx + k3vx)
        k4y = dt * (vy + k3vy)
        k4z = dt * (vz + k3vz)
        k4vx = dt * ax4
        k4vy = dt * ay4
        k4vz = dt * az4

        x_new = x + (1 / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y_new = y + (1 / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
        z_new = z + (1 / 6) * (k1z + 2 * k2z + 2 * k3z + k4z)
        vx_new = vx + (1 / 6) * (k1vx + 2 * k2vx + 2 * k3vx + k4vx)
        vy_new = vy + (1 / 6) * (k1vy + 2 * k2vy + 2 * k3vy + k4vy)
        vz_new = vz + (1 / 6) * (k1vz + 2 * k2vz + 2 * k3vz + k4vz)

        x_values.append(x_new)
        y_values.append(y_new)
        z_values.append(z_new)
        vx_values.append(vx_new)
        vy_values.append(vy_new)
        vz_values.append(vz_new)
        los.append(np.tan(alpha)*x_values[-1])
        if error_method == "SEM":
            yerror+=(y_values[-1]*abs(x_values[-2]-x_values[-1]))**2
            zerror+=((z_values[-1]-los[-1])*abs(x_values[-2]-x_values[-1]))**2
        elif error_method =="TEM":
            yerror+=abs(y_values[-1]*abs(x_values[-2]-x_values[-1]))
            zerror+=abs((z_values[-1]-los[-1])*abs(x_values[-2]-x_values[-1]))
        else:
            yerror+=(y_values[-1]*abs(x_values[-2]-x_values[-1]))
            zerror+=((z_values[-1]-los[-1])*abs(x_values[-2]-x_values[-1]))
            
    squared_error=yerror**2+zerror**2

    return x_values, y_values, z_values, vx_values, vy_values, vz_values, los, squared_error

def acceleration(velocity):
    v_mag = np.linalg.norm(velocity)
    ax,ay,az=-k*velocity*v_mag
    return ax, ay, az-g
    
    
def initial_position(yf,zf,alpha,beta):
    y0=-yf*np.sqrt(yf**2+l1**2)/l1
    z0=-(zf*l1-l2*(hr-zf))/(np.sqrt((hr-zf)**2+l1**2))
    p0temp=roll(beta)@np.array([0,y0,z0])

    return p0temp[1], p0temp[2]/np.cos(alpha)
    
    
def initial_velocity(yf,zf,alpha,beta):
    s_vec=s_vector(yf,zf,alpha,beta)
    s_mag=np.linalg.norm(s_vec)
    return s_vec*v0/s_mag
    
def s_vector(yf,zf,alpha,beta):
    sx=l1
    sy=-yf
    sz=hr-zf
    s_vec=pitch(alpha) @ roll(beta) @ np.array([sx,sy,sz])
    return s_vec
    
def pitch(alpha):
    R=np.array([[np.cos(alpha),0,-np.sin(alpha)],
        [0,1,0],
        [np.sin(alpha),0,np.cos(alpha)]])
    return R

def roll(beta):
    R=np.array([[1,0,0],
        [0,np.cos(beta),-np.sin(beta)],
        [0,np.sin(beta),np.cos(beta)]])
    return R
    
    
    