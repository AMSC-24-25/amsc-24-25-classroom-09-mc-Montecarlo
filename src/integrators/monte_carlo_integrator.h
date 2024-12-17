#ifndef MONTE_CARLO_INTEGRATOR_H
#define MONTE_CARLO_INTEGRATOR_H

#include "domain/integration_domain.h"
#include <random>
#include <functional>
#include <cstddef>
#include <vector>

// MonteCarloIntegrator class, which performs Monte Carlo integration over a given integration domain
class MonteCarloIntegrator {
    // Our domain of integration
    const IntegrationDomain &domain;

    // Each distribution generates random numbers in a specific range, different for each dimension
    std::vector<std::uniform_real_distribution<double>> distributions;

    // Random engines, mt19937 shows the best performance/quality balance
    std::vector<std::mt19937> engines;

    // Initializes pseudo-random engines with seeds from std::seed_seq
    void initializeEngines(size_t numThreads);

    // Returns an n-dimensional point with pseudo-random independent coordinates
    std::vector<double> generatePoint(std::mt19937 &eng);

public:
    // Constructor that sets up the distributions based on domain bounds
    explicit MonteCarloIntegrator(const IntegrationDomain &d);

    // Integrates real-valued function `f` over the domain specified in the constructor
    double integrate(
        const std::function<double(const std::vector<double> &)> &f,
        size_t numPoints,
        size_t numThreads
    );

    // Stratified integration method
    // f: function we want to integrate
    // numPoints: total number of points generated for integration
    // numThreads: number of threads used for parallelization
    // strataPerDimension: number of strata (subdivisions) in each dimension
    double integrateStratified(
        const std::function<double(const std::vector<double> &)> &f,
        size_t numPoints,
        size_t numThreads,
        int32_t strataPerDimension
    );
};

#endif // MONTE_CARLO_INTEGRATOR_H
