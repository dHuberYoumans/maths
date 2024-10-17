# MeanCurvatureFlow

**Work in progress**

## TO DO
* define proper triangulations and their refinements
* implement Platonic solids

## Abstract

This small project is about minimal surfaces and the mean curvature flow of different shapes. 

``src/`` contains the scripts

* ``geometry.py`` which defines several methods such as sub-dividing a triangle of computing edges and faces from a given set of vertices
* ``geometric_objects`` which defines several geometric objects (e.g. a spherical mesh obtained from an icosahedron - an icosphere)

## geometric_objects

The geometric objects (shapes) whose mean curvature flow we want to study are defined in terms of a uniform mesh. 
They are strored in ``vertices`` (list of vertices in 3D space) ``edges`` (set of 2-tuples---each tuple contains the two indices of the adjacent vertices) and ``faces`` (set of 3-tuples---each tuple contains the three indices of the adjacent vertices)
