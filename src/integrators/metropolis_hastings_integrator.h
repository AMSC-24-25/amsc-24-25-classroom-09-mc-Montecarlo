#ifndef METROPOLIS_HASTINGS_INTEGRATOR_H
#define METROPOLIS_HASTINGS_INTEGRATOR_H

#include <random>
#include <functional>
#include <vector>
#include <future>
#include <iostream>

#include "domain/integration_domain.h"

/*
Metropolis-Hastings is useful to get a sample generated from a "complex" probability distribution.
How does it work:
    - choose a starting point x0 (we can choose whatever we want, like the origin)
    - define a proposal distribution Q(x'|x), it will generate x' starting from x
    - x' is the new candidate point
    - compute r (acceptance ratio)
    - decide to accept the point or not
    - repeat
To parallelize it: divide the sample into multiple chains and compute the avg
*/

// Metropolis-Hastings Integrator class with parallel chains
class MetropolisHastingsIntegrator {
    const IntegrationDomain &domain;
    std::normal_distribution<double> proposalDist; // Gaussian proposal distribution

    // Initializes random engines with unique seeds for each chain
    std::vector<std::mt19937> initializeEngines(size_t numChains);

public:
    // Constructor to initialize the integrator with a domain
    explicit MetropolisHastingsIntegrator(const IntegrationDomain &d);

    // Perform a single Metropolis-Hastings chain
    // Returns a pair of {result, #accepted points}
    std::pair<double, int32_t> integrateSingleChain(
        const std::function<double(const std::vector<double> &)> &f,
        size_t numPoints,
        const std::vector<double> &initialPoint,
        std::mt19937 &engine
    );

    // Perform parallel integration with multiple chains
    // Returns a pair of {result, acceptance rate}
    std::pair<double, double> integrateParallel(
        const std::function<double(const std::vector<double> &)> &f,
        size_t numPoints,
        const std::vector<double> &initialPoint,
        size_t numChains
    );
};

#endif // METROPOLIS_HASTINGS_INTEGRATOR_H
