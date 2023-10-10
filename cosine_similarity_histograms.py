# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:52:44 2023

@author: thomas.kendall
"""
import random as random
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
 

def get_not_random_vector(k):
    vec=np.zeros(k)
    vec[0]=1
    vec[k-1]=.5
    for i in range(k):
        vec[i]=max(min(1,vec[i]+((-1)**random.randint(0, 1))*random.random()*.1),0)
    return vec        

def get_random_vector(k):
    
    vec=np.array([random.random() for i in range(k)])
    return vec

def cosine_similarity(vec1, vec2):
    
    vec1_unit=vec1/np.linalg.norm(vec1)
    vec2_unit=vec2/np.linalg.norm(vec2)
    return vec1_unit @ vec2_unit

def plot_cosine_similarities(k,N):
    plt.clf()
    cosines_random =[cosine_similarity(get_random_vector(k),get_random_vector(k)) for i in range(N)]
    cosines_not_random =[cosine_similarity(get_not_random_vector(k),get_not_random_vector(k)) for i in range(N)]
    fig, axs = plt.subplots(2)
    axs[0].hist(cosines_random, bins = 10)
    axs[1].hist(cosines_not_random, bins = 10)
    for i in range(2):
        axs[i].set_xlim(0,1)
    axs[0].set_title("Cosine Simularity Distribution Between Random Vectors")
    axs[1].set_title("Cosine Simularity Distribution Between Non-Random Vectors")
    return