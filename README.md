# Monte Carlo Integration

This project demonstrates parallel numerical integration of functions over various geometric domains using Monte Carlo
methods. It includes implementations of standard Monte Carlo integration, stratified sampling, and Metropolis-Hastings
sampling.

## Understanding Monte Carlo Integration

Monte Carlo integration is a technique for approximating integrals by using random samples. Instead of analytically
computing the integral, we estimate it by drawing points uniformly from the domain and evaluating the integrand at these
points. As the number of samples increases, the approximation typically converges to the true value of the integral.

The basic formula for Monte Carlo integration is:

$$\int_{\Omega} f(x)\,dx \approx \frac{|\Omega|}{n} \sum_{i=0}^{n-1} f(x_i)$$

- $\Omega$ is the domain over which we integrate.
- $f(x)$ is the function to be integrated.
- $|\Omega|$ denotes the measure (volume, area, length) of the domain.
- $x_i$ are points sampled uniformly at random from \( \Omega \).
- $n$ is the number of sample points.

The accuracy of the Monte Carlo approximation improves with the number of samples, and various variance-reduction
techniques (such as stratified sampling) can be applied.

## Sampling Methods

### Standard Monte Carlo (Uniform Sampling)

In standard Monte Carlo integration, we:

1. Determine a bounding region that encompasses the entire domain $\Omega$.
2. Draw points uniformly at random from this bounding region.
3. Evaluate $f$ at each point. If a point falls outside $\Omega$, its contribution is zero.
4. Average these values and multiply by the measure of the bounding region.

This approach is straightforward but may have high variance, especially if $\Omega$ is complex or if $f$ varies
significantly within the domain. Uniform sampling does not focus on any particular region of the domain and treats all
areas equally, which can lead to inefficient sampling, particularly in higher dimensions or for irregular shapes.

### Stratified Sampling

Stratified sampling reduces variance by partitioning the bounding region into smaller, non-overlapping subregions
(strata) and sampling from each one. By ensuring that every stratum is represented in the sample set, this method:

1. Provides a more uniform coverage of the domain.
2. Reduces the likelihood of clustering samples in just a few areas.
3. Improves the accuracy of the integral estimate with fewer samples compared to plain uniform sampling.

In practice, you divide each dimension into several intervals to form a grid of strata. For each stratum, you take at
least one random sample. The result is then a weighted average of all strata. Since each subregion is sampled, the
overall
estimate tends to have lower variance, making stratified sampling more efficient and often more accurate for the same
number of samples.

### Metropolis-Hastings (MCMC)

Metropolis-Hastings is a Markov chain Monte Carlo (MCMC) technique typically used to sample from complex probability
distributions. Instead of sampling independently and uniformly, Metropolis-Hastings builds a chain of samples where each
new sample is proposed based on the current state, and accepted or rejected according to an acceptance ratio derived
from
the target distribution.

For integration purposes, if the target distribution is related to the integrand $f(x)$ (for instance, a probability
distribution proportional to $f(x)$ if $f$ is non-negative), then Metropolis-Hastings can focus sampling on the
regions that matter most. This can lead to more efficient exploration of complex domains or integrands, especially where
uniform sampling would waste many samples in low-interest areas.

However, using Metropolis-Hastings for uniform integration requires adjustments to ensure that the distribution of
samples matches the uniform density over the domain. This might be less straightforward than standard or stratified
sampling, but it can be beneficial for challenging or high-dimensional integrals.

## Features

- **Parallel Implementation**:
    - The integration process is parallelized by splitting the total number of samples (`n`) across multiple threads.
    - Each thread independently generates and evaluates its own subset of points and accumulates partial sums.
    - After all threads finish, their partial sums are combined (reduced) to produce the final integral estimate. This
      leverages multiple CPU cores to speed up the computation.

- **Multiple Integration Domains**:
    - **Hypersphere**: A hypersphere of arbitrary dimension and radius, bounded by a hypercube.
    - **Polygon2D**: A polygon in 2D, defined by a set of vertices.
    - **Polytope** (H-form): A convex polytope defined by linear inequalities (Ax ≤ b) and an explicit bounding box.

## Directory Structure

- **`domain/`**: Contains classes representing geometric domains (`Hypersphere`, `Polygon2D`, `Polytope`) and the
  `IntegrationDomain` interface they all implement.
- **`integrators/`**: Contains integration strategies: `MonteCarloIntegrator` for standard and stratified sampling, and
  `MetropolisHastingsIntegrator` for MH-sampling.
- **`main.cpp`**: Demonstrates integrating different functions over different domains.

## Dependencies

- **C++17 or later**:
- **CMake**: For building the project.
- **Eigen**: A C++ template library for linear algebra.

## Building and running the project

1. Ensure you have CMake and C++17 or newer installed.
2. Clone the repository:
   ```bash
   git clone git@github.com:AMSC-24-25/09-mc-09-mc.git
   cd 09-mc
   ```
3. Create a build directory and run CMake:
   ```bash
    mkdir build && cd build
    cmake ..
    make
   ```
4. Run:
    ```bash
    ./09-mc
    ```

The program shows:

- Integrating `f(x,y) = x² + y²` over a unit circle, displaying results from standard, stratified, and
  Metropolis-Hastings
  methods for various numbers of points.
- Integrating `f(x,y) = 1` over an equilateral triangle (using `Polygon2D`) and print a similar comparison table.
- Optionally, you can integrate over a `Polytope` defined in `polytope.h/cpp`.

## Possible Improvements

- **Physics application**: Show a real-life physics scenario where monte-carlo integration can be used.
- **Advanced domains**: Add more complex domains or integrate with convex polytope libraries.
- **Variance Reduction**: Enhance stratification or add Latin hypercube sampling for even better variance reduction.
