# Mathematics for the eye and other senses

The following projects are result of tinkering around with funny ideas about how to nicely present mathematical concepts and how to implement them in code.
The projects highlight the (maybe not so surprising) similarities of concepts of pure mathematics and machine learning. Albeit mathematics is the basic language for computer science and programming, the approach to a given problem in pure maths (at least in some areas) often diverges from that of, say, computer science. Coming from mathematical physics, I'd like to take a step through the looking glas and compare, here and there, machine learning techniques with their purely mathematical mirror images.

##### Table of Contents
- [Getting Started](#getting-started)  
- [Prerequisites](#prerequisites)  
- [The Projects](#the-projects)  
    * [ImgSampling](#imgsampling)
    * [MeanCurvFlow](#meancurvflow)
    * [Conway's Game of Life](#conway-s-game-of-life)

## Getting Started 

Most of the projects are either given in terms of a Python script possibily in addition to a Jupyter notebook showcasing how to work with the script.
Moreover, each project will have its own folder containing a README.md for more details on the idea behind the project. Just head on over and check them out!

## Prerequisites

Explaining the theory of each mathematical concept thoroughly is beyond the scope of the projects. We will, however, give a small outlook of the key-ideas. Familiarity with basic geometry, statistics and physics would of course be beneficial, is, however, not strictly needed.

## The Projects

### ImgSampling

This project arises from a simple question: how does one randomly draw samples from a given non-uniform distribution? It is relatively easy to draw randomly from a uniform distribution and many programming languages have built in mehtods for this.

For example, in Python one could use `numpy.random.uniform` to draw from a uniform distribution

```
import numpy as np

x = np.random.uniform(0, 1)
x
```

To draw from _any_ other distribution $f(x)$, one can exploit the easiness of drawing uniformly by first computing the associated *cumulative distribution function* (CDF) 

$$F(x) = \int_{-\infty}^x f(x)dx$$

Once one knows the CDF, if $Y$ is a random variable which is uniformly distributed, the random variable $X = F^{-1}(Y)$ will be distributed according to $f(x)$. This method is known as _inverse transform sampling_.

The projects now puts a twist on this idea: it lets the user load an image, which is converted into gray scale (using values between 0 = black and 1 = white).
The image, which is essentially just a 2D array, can be interpreted as a 2D distribution funtion after normalising it by the sum of all its vlaues.

We then sample points according to the distribution defined by the input image using the inverse transform sampling. In order to see if we sample correctly, we plot the randomly drawn points in a scatter plot. The color of each point is represented by its probability to being randomly drawn. If we did everything correctly, we should get back (an impressionistic interpretation of) our original image!

**Example** 

(left: original image, right: sampled image)

<p align="center">
  <p align="left" >
    <img align="left" float="left"  src="https://github.com/dHuberYoumans/maths/blob/main/images/Eddie.jpeg" title="original image" height=400px width=auto />
  </p>
  <img align="middle" float="left" src="https://github.com/dHuberYoumans/maths/blob/main/images/Eddie_sampled.png" title="sampeld image" height=400px width=auto />
</p>

**Remark** There are other sampling algotithms, for example _rejection sampling_ (see the [documentation](https://github.com/dHuberYoumans/maths/blob/main/ImgSampling/README.md)) for more information), which we implement as well.

### MeanCurvFlow

This project is about "minimal surfaces". In mathematics, one way to definition states that a surfaces with vanishing _mean curvature_ is called a _minimal surface_.
The concept of mean curvature is easier than it sounds: firstly, if one considers a curve in the plane (think of a curve -- like a circle -- in the classic coordinate system with $x$- and $y$-axis) one way to measure its _curvature_ is by the inverse of the radius of the circle which best approximates the curve at this point (for more details see the README of the project). 

Now, given a _smooth_ (that is without corners or sharp edges) surface embedded in 3D, $\Sigma \subset \mathbb{R}^3$, like the sphere, or a cylinder or the surface of the ocean, one can associate to each point a _tangent plane_, which is a 2D plane which touches the surface at the chosen point and nowhere else. This plane has a unique _normal vector_, a vector which is everywhere perpendicular to that plane. The collection of all those normal vectors is called the _normal vector field_ of the surface. 

Suppose thus we are given a surface $\Sigma \subset \mathbb{R}^3$ and a point $p \in \Sigma$ of the surface $\Sigma$. Let us denote by $\vec N(p)$ the normal vector at that point. Then we can consider a plane $E$ which contains $\vec N(p)$. This plane $E$ intersects the surfaces $\Sigma$ and the points of intersection form a curve on that plane. For example, if $\Sigma$ is a sphere in 3D and the point $p$ its north pole, the normal vector $\vec N(p)$ points straight up along the $z$-axis and any plane $E$ which contains $\vec N(p)$ intersects the sphere in a "great circle" (it is instructive to take a minute to think about this claim). 

But now we have a curve and we know how to compute its curvature! Call it $\kappa$. But of course there are many planes containing $\vec N(p)$. In fact, starting with any one plane $E_0 = E(0)$, we can rotate it around the axis defined by $\vec N(p)$. Thus, for the family of planes $E(\theta)$ (parametrised by an angle $\theta$) we can define a family of curvatures $\kappa(\theta)$. And one defines the _mean curvature_ just as the mean of all those curvatures 

$$H_p = \frac{1}{2\pi} \int_0^{2\pi} \kappa(\theta) d\theta$$

(See the documentation of the project for a nice link to _principal component analysis_ (PCA) from machine learning) 
And a _minimal surface_ is now defined by the condition that $H_p = 0$ for all points $p$ of the surface.

In order to study such surfaces, in this project, we define a surface in a discretised mannar, namely by a _triangulation_.
A triangulation consists of vertices, edges (connecting these vertices) and faces (triangles -- hence the name triangulation -- formed by these edges).  
A base class `GeometricObject3D()` defines basic methods such as getters for the collection of vertices, edges and faces but also more involved plotting methods.
The actual surfaces are defined by classes which inherit from this base class

```
class Cylinder(GeometricObject3D):
  def __init__(self,steps:int = 10, radius:float = 1.0, height: float = 1.0):
    # define triangulation of the cyliner
```

Once defined, one can then plot the surface in various ways, for example as a _cloud of points_, its _triangulation_ or its _normals_

```
cylinder = geomObj.Cylinder(steps=2)

cylinder.plot_cloud_of_pts(title='surface as cloud of points')
cylinder.plot_mesh(v_color='k',e_color='gray',title=f'triangulation (filled faces)',fill=True)
cylinder.plot_normals(scale=0.3,title='normals with mesh and filled faces',mesh=True,fill=True)
```

<p align="center">
     <img align="left" float="left" src="https://github.com/dHuberYoumans/maths/blob/main/images/cylinder_cloud_of_pts.png" title="cloud of points" height=300px width=auto />
     <img align="left" float="left" src="https://github.com/dHuberYoumans/maths/blob/main/images/cylinder_mesh_filled.png" title="triangulation" height=300px width=auto />
      <img align="left" float="left" src="https://github.com/dHuberYoumans/maths/blob/main/images/cylinder_normals.png" title="normals" height=300px width=auto />
</p>

## Conway's Game of Life
This project is an implementation of the Conway's famous *Game of Life*.

<p align="center">
  <img src="https://github.com/dHuberYoumans/maths/blob/main/images/GoL_Toad.gif" alt="animated" width=500px height=auto />
</p>

The rules are simple: The earth is a flat, quadrilateralized torus---a doughnut modeled by a grid of squares with periodic boundary conditions; going out one side one enters again from the opposite side. The classic _Pac-Man_ or _Snake_ scenario.

Each square represents a cell, which can be either _alive_ or _dead_. 
The collection of squares is called a _generation_ and passing to the next generation cells can either stay alive or die (for example due to over- or under-population). But they can also be (re)born! When a cell dies or is born depends on its 8 neighbours. 

There is a lot of interesting maths behind this beautifully simple game: the theory of **cellular automata**.

To play around with the project, simply run ```main.py``` with Python (tested with Python v.3.11).
It will prompt our little world as a 20x20 grid (the size can be adjusted by modifing ```main.py```).
To generate a population, simply click on those cells which should live (_white_ means live, _black_ means death) and hit the _Play_ button. 
If you want to pause to study or modify the current population, simply press the _Pause_ botton and again the _Play_ button to continue. 
Finally, one can reset the world with the _Reset_ button.

Have fun!


