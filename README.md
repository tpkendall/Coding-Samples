# Coding Samples
 Examples of Coding
These examples are pulled from a much larger collection of coding.

Visual Field Explainer.nb 
  - Mathematica Notebook which demonstrates the multivariable calculus used to determine the probability of detecting movement visually.

cosine_similarity_histograms.py 
  - Python code used to refamiliarize with cosine similarity. 
  - The function plot_cosine_similarities(5,10000) generates 10000 randomly generated vectors of length 5 and 10000 not so randomly generated vectors of length 5 and shows the distribution of cosine values.

detection_avoidance.py
  - This code is used to generate visualizations, arc lists, and node fields for my optimization problem of fiding the path of least detection in the presence of enemies.
  - Currently the use inputs suspected enemy locations, orientation, and certainties about those values.
  - The map is then overlaid with a node field of three layers corresponding to movement methods
  - The code then generates probability of detection between adjacent nodes using various measures of probability
  - A different code in Julia would then find the path of least detection from a user generated start point and end point.

generate_river_nodes_intersections.py
  - Written quickly to generate scenarios to optimize contested logistic routes.
  - Generates a random river
  - Generates a user specified number of intersections
  - Randomly designates intersection as rail or air according to a set probability of occurance.
  - Designates intersections that occur in the river as water access points and moves them to the edge of the river
  - connects nearest neighbors randomly according to type
  - Outputs arcs for optimizing over

smart_sight_adjustments.py
  - Used for programming a smart sight that accounts for the holder's tilt and roll
  - Uses 4th order runge-kutta to integrate error of trajectory compared to line of sight out to 300m
  - minimizes the net error in two dimensions using a custom "tightening net search"
  - can be run for any tilt roll angle combination
