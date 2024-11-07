# ImgSampling

This project allows the user to upload an image and interprete them as 2D probability densities. One can then draw samples randomly from this probability density. To check if the algoritm is doing the correct thing, one can create a scatter plot of the randomly chosen samples painting them according to their probability. The result is an impressionistic probabilistic interpreation of the original image.

##### Table of Contents
- [Theory](#theory)
   * [The uniform distribution pdf and cdf](#the-uniform-distribution-pdf-and-cdf)
   * [Sampling any pdf](#sampling-any-pdf)
   * [Example](#example)
 
  
 
## Theory

### The uniform distribution pdf and cdf
Before diving into the inner workings of the algorithm, let us quickly look at the mathematical theory behind it. 
The main question we want to adress is _how can we draw samples from a given probability distribution_? 

Let us start with an example.
A _uniform_ distribution is a probability distribution where each point inside a certain domain, say an interval $[a,b]$ is the same. 

For example, consider the throw of a dice. Each face $1,2,\dots,6$ has the same probability, namely $1/6$. 
Mathematically we can write this as follows: the throw of a dice is considered to be a _random variable_ $X$. It is a function on the sample space---the space of possible outcomes of throwing the dice---$S = \{1,2,\dots,6 \}$. For the experiment "throwing a dice", the likelihoods $P(X = s)$ (read: the probability of $X$ taking the value $s$) of throwing $s = 1,2,3,\dots$ is given by the assignment $P(X = s) = 1/6$ for _any_ outcome $s \in S$. Since the likelihoods are all the same, the probability _distribution_ $P(X)$ is called _uniform_. Note that $P(X = s)$ is just $1/|S|$ where $|S|$ stands for the number of elements in the set $S$ (here $|S| = 6$).

We could also think of a continous setting where the the dice could have a "face of any real number" inside an interval $[a,b] \subset (-\infty,\infty)$. This interval is again the sample space, i.e. the space of possible outcomes, of the experiment "X: throwing a dice".
In this case, it is more apropriate to think about the the probability _density_ function (PDF)

$$
\begin{align*} 
p(x) = \begin{cases} \frac{1}{b - a} \quad &\text{if~} x \in [a,b] \\\\ \\\\ 0 \quad &\text{otherwise} \end{cases} 
\end{align*}
$$

<p align="center">
 <img align="top"  src="https://github.com/dHuberYoumans/maths/tree/main/ImgSampling/img_README/uniform_pdf.png" title="uniform distribution" height=300px width=auto />
</p>

In the continous case, the probability of $X$ being smaller or equal to a certian given number $x$ is given by the cumulative distribution function (cdf) represented by the integral

$$ F_X(x) = P(X \leq x) = \int_{-\infty}^x p(x) dx $$

<p align="center">
 <img align="top"  src="https://github.com/dHuberYoumans/maths/tree/main/ImgSampling/img_README/uniform_cdf.png" title="uniform distribution" height=300px width=auto />
</p>

And due to the linear properties of the integral, we can compute the probability to throw a number $x$ such that $a < x < b$ simply by considering 

$$P(a < x < b) = F_X(b) - F_X(a) = \int_{-\infty}^b p(x)dx - \int_{-\infty}^a p(x)dx = \int_a^b p(x)dx$$

 The cdfs have to respect some natural conditions in order to allow an interpretation as probabilities:

 * domain: the cdf maps the real number line to the interval $[0,1]$: $F_X(x)\colon \mathbb{R} \to [0,1]$
   
 * positivity: for any $x$, $F_X(x) \geq 0$ (indeed it does not make a lot of sense to talk about negative probabilities)
   
 * normalisation: $\lim_{x\to \infty} F_X(x) = 1$ (certainly, if $F_X(x) = P(X \leq x)$ measures the probability that when we are thrwoing our infinite dice (whose faces can be any real number $x$) then certainly we will find some $x < \infty$) 

### Sampling any pdf 
For a uniform distribution, there is an interesting fact, which we will exploit in a minute.
Suppse that $U$ is a random variable that is uniformly distributed on $[0,1]$, that is its pdf is of the form 

$$
\begin{align*} 
p(u) = \begin{cases} 1 \quad &\text{if~} x \in [0,1] \\\\ \\\\ 0 \quad &\text{otherwise} \end{cases} 
\end{align*}
$$

How does its cdf look like? If one computes the integral, one finds a surprisingly simple form! Namely 

$$
\begin{align*} 
F_U(u) = \begin{cases} u \quad &\text{if~} u \in [0,1] \\\\ \\\\ 0 \quad &\text{otherwise} \end{cases} 
\end{align*}
$$

The cdf is just linear (see the picture above) inside the interval and zero everywhere else!

This linear relationship is what allows us to find the pdf starting from a known cdf! Let's pause and think about this for a minute. Suppose that we are given a cdf $F(x)$ (a function which simply satisfies the above conditions of having the correct domain, a correct normalisation and a correct limit as $x\to\infty$). A natural question to ask would be what is the associated pdf $p(x)$? We might be tempted to simply take the derivative and indeed if $F(x)$ is _nice enough_ that would be indeed true. However, there is another way to compute it. Suppose that $U$ is uniformally distributed. Then we claim that the random variable given by the inverse (in an appropriate sense) of $F$, $X = F^{-1}(U)$, is distributed according to the cdf $F(x)$. Let us start with this assumption and show that it is true. By the definition of a cdf, we have

$$P(X \leq x) = P( F^{-1}(U) \leq x) = P( U \leq F(x) )$$

But since $U$ is uniformly distributed, $P( U \leq F(x)) = F(x)$ and so we find 

$$P( X \leq x) = F(x)$$

which means that $F(x)$ is indeed the cdf of the random variable $X = F^{-1}(U)$.

But this is a neat result! In particular, because many programming languages have built in methods to randomly sample uniformally distributed numbers which now allow us to sample according to _any_ distribution we like! Here is the algorithm: suppose we want to draw numbers (or throw dices whose faces are) distributed according to some pdf $p(x)$. Then

1. find the cdf $F_X$
2. find its inverse $F_X^{-1}$
3. draw / throw a uniformly distributed random number $u$ using built in methods
4. the number $x = F_X^{-1}(u)$ is a random sample from the orighinal distribution $p(x)$!
 
**Remark** Actual randomness is hard to achieve and to the best of my knowledge still an open problem. This is why many sources (rightfully) speak of _pseudo_ random numbers when they are generated by the computer.

### Example

