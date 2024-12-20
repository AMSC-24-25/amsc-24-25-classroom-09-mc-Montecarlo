#ifndef METROPOLIS_HASTINGS_INTEGRATOR_H
#define METROPOLIS_HASTINGS_INTEGRATOR_H

#include "abstract_integrator.h"
#include "domain/integration_domain.h"

#include <random>
#include <functional>
#include <vector>
#include <future>

/*
Metropolis-Hastings algorithm for sampling from a complex probability distribution p(x)
and computing expectations E[h(x)] with respect to this distribution.

How does it work:
    - choose a starting point x0
    - define a proposal distribution Q(x'|x) (here: Gaussian centered at current point)
    - generate new candidate point x' from Q(x'|x)
    - compute acceptance ratio r = p(x')/p(x)
    - accept x' with probability min(1,r)
    - for accepted points, accumulate h(x) to compute E[h(x)]
    - repeat

The algorithm will generate samples distributed according to p(x),
which lets us compute E[h(x)] = ∫h(x)p(x)dx / ∫p(x)dx.

To parallelize: run multiple independent chains and average their results
*/

// Metropolis-Hastings sampler class with parallel chains
class MetropolisHastingsIntegrator : public AbstractIntegrator {
    // Gaussian noise distribution
    std::normal_distribution<double> proposalDist;

public:
    // Constructor to initialize the sampler with a domain and stddev of the Gaussian noise
    explicit MetropolisHastingsIntegrator(const IntegrationDomain &d, double stddev);

    // Perform a single Metropolis-Hastings chain to compute E[h(x)] with respect to p(x)
    // Returns a pair of {E[h(x)], #accepted points}
    std::pair<double, int32_t> integrateSingleChain(
        const std::function<double(const std::vector<double> &)> &f, // Function to get expectation of
        const std::function<double(const std::vector<double> &)> &p, // Target distribution
        size_t numPoints,
        const std::vector<double> &initialPoint,
        std::mt19937 &engine
    );

    // Perform parallel computation of E[h(x)] using multiple chains
    // Returns a pair of {E[h(x)], acceptance rate}
    std::pair<double, double> integrateParallel(
        const std::function<double(const std::vector<double> &)> &f,
        const std::function<double(const std::vector<double> &)> &p,
        size_t numPoints,
        const std::vector<double> &initialPoint,
        size_t numChains
    );
};

#endif // METROPOLIS_HASTINGS_INTEGRATOR_H
