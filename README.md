# Mathematics for the eye and other senses

The following projects are result of tinkering around with funny ideas about how to nicely present mathematical concepts and how to implement them in code.
The projects highlight the (maybe not so surprising) similarities of concepts of pure mathematics and machine learning. Albeit mathematics is the basic language for computer science and programming, the approach to a given problem in pure maths (at least in some areas) often diverges from that of, say, computer science. Coming from mathematical physics, I'd like to take a step through the looking glas and compare, here and there, machine learning techniques with their purely mathematical mirror images.

## Getting Started

Most of the projects are either given in terms of a Python script possibily in addition to a Jupyter notebook showcasing how to work with the script.
Moreover, each project will have its own folder containing a README.md for more details on the idea behind the project. Just head on over and check them out!

### Prerequisites

Explaining the theory of each mathematical concept thoroughly is beyond the scope of the projects. We will, however, give a small outlook of the key-ideas. Familiarity with basic geometry, statistics and physics would of course be beneficial, is, however, not strictly needed.

## The Projects

### ImgSampling

This project arises from a simple question: how does one randomly draw samples from a given non-uniform distribution? It is relatively easy to draw randomly from a uniform distribution and many programming languages have built in mehtods for this.

For example, in Python one could use ``numpy.random.uniform`` to draw from a uniform distribution

```
import numpy as np

x = np.random.uniform(0, 1)
x
```

To draw from *any* other distribution $f(x)$, one can exploit the easiness of drawing uniformly by first computing the associated *cumulative distribution function* (CDF) $$F(x) = \int_{-\infty}^x f(x)dx$$. Once one knows the CDF, if $Y$ is a random variable which is uniformly distributed, the random variable $X = F^{-1}(Y)$ will be distributed according to $f(x)$. This method is known as *inverse transform sampling*.

The projects now puts a twist on this idea: it lets the user load an image, which is converted into gray scale (using values between 0 = black and 1 = white).
The image, which is essentially just a 2D array, can be interpreted as a 2D distribution funtion after normalising it by the sum of all its vlaues.

We then sample points according to the distribution defined by the input image using the inverse transform sampling. In order to see if we sample correctly, we plot the randomly drawn points in a scatter plot. The color of each point is represented by its probability to being randomly drawn. If we did everything correctly, we should get back (an impressionistic interpretation of) our original image!


### MeanCurvFlow

