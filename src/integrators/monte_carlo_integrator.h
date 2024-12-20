#ifndef MONTE_CARLO_INTEGRATOR_H
#define MONTE_CARLO_INTEGRATOR_H

#include "abstract_integrator.h"

#include <functional>
#include <cstddef>
#include <vector>

// MonteCarloIntegrator class, which performs Monte Carlo integration over a given integration domain
class MonteCarloIntegrator : public AbstractIntegrator {
    // Each distribution generates random numbers in a specific range, different for each dimension
    std::vector<std::uniform_real_distribution<double>> distributions;

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
    double integrateStratified(
        const std::function<double(const std::vector<double> &)> &f,
        size_t numPoints,
        size_t numThreads,
        int32_t strataPerDimension
    );
};

#endif // MONTE_CARLO_INTEGRATOR_H
