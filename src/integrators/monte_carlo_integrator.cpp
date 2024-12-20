#include "monte_carlo_integrator.h"

#include <future>
#include <numeric>
#include <chrono>
#include <thread>

MonteCarloIntegrator::MonteCarloIntegrator(const IntegrationDomain &d) : AbstractIntegrator(d) {
    // Create a random-value uniform distribution in [min, max] for each dimension using the domain bounds.
    auto bounds = domain.getBounds();
    for (const auto &[min, max]: bounds) {
        distributions.emplace_back(min, max);
    }
}

std::vector<double> MonteCarloIntegrator::generatePoint(std::mt19937 &eng) {
    std::vector<double> point(distributions.size());
    for (size_t i = 0; i < distributions.size(); ++i) {
        point[i] = distributions[i](eng);
    }
    return point;
}

double MonteCarloIntegrator::integrate(
    const std::function<double(const std::vector<double> &)> &f,
    size_t numPoints,
    size_t numThreads
) {
    initializeEngines(numThreads);
    size_t pointsPerThread = numPoints / numThreads;

    // Represents the work of a single thread
    auto worker = [this, &f, pointsPerThread](size_t myIndex) {
        double sum = 0.0;

        for (size_t i = 0; i < pointsPerThread; ++i) {
            auto point = generatePoint(engines[myIndex]);

            // If the point is in the domain, evaluate the function f and add the value to our partial result.
            if (domain.contains(point)) {
                sum += f(point);
            }
        }

        return sum;
    };

    std::vector<std::future<double>> futures;

    // Start numThreads worker threads
    futures.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        futures.push_back(
            std::async(std::launch::async, worker, i)
        );
    }

    // Retrieve partial results from each worker thread and combine them into a single total result.
    double totalSum = 0.0;
    for (auto &future: futures) {
        totalSum += future.get();
    }

    return totalSum * domain.getBoundedVolume() / static_cast<double>(numPoints);
}

double MonteCarloIntegrator::integrateStratified(
    const std::function<double(const std::vector<double> &)> &f,
    size_t numPoints,
    size_t numThreads,
    int32_t strataPerDimension
) {
    initializeEngines(numThreads);
    size_t pointsPerThread = numPoints / numThreads;

    // Number of dimensions
    size_t dimensionCount = distributions.size();

    // Total number of strata in the domain = strataPerDimension^dimensionCount
    size_t totalStrata = 1;
    for (size_t i = 0; i < dimensionCount; ++i) {
        totalStrata *= strataPerDimension;
    }

    size_t samplesPerStratum = pointsPerThread / totalStrata;

    // Helper function to convert a linear stratum index to multi-dimensional indices
    auto decodeStratumIndex = [&](size_t linearIndex) {
        // Convert linearIndex to base strataPerDimension in dimensionCount dimensions
        std::vector<int32_t> strataIndices(dimensionCount, 0);
        for (int d = static_cast<int>(dimensionCount) - 1; d >= 0; --d) {
            strataIndices[d] = static_cast<int32_t>(linearIndex % strataPerDimension);
            linearIndex /= strataPerDimension;
        }
        return strataIndices;
    };

    // Worker lambda: systematically iterate over all strata
    auto worker = [this, &f, samplesPerStratum, strataPerDimension, totalStrata, &decodeStratumIndex](size_t myIndex) {
        double sum = 0.0;
        std::mt19937 &eng = engines[myIndex];

        for (size_t stratumId = 0; stratumId < totalStrata; ++stratumId) {
            // Get the strata indices for each dimension
            auto strataIndices = decodeStratumIndex(stratumId);

            // For each dimension, calculate the lower and upper bounds of this stratum
            std::vector<std::uniform_real_distribution<double>> strataDists;
            strataDists.reserve(distributions.size());
            for (size_t dim = 0; dim < distributions.size(); ++dim) {
                double a = distributions[dim].a();
                double b = distributions[dim].b();
                double step = (b - a) / strataPerDimension;
                double lower = a + strataIndices[dim] * step;
                double upper = lower + step;
                strataDists.emplace_back(lower, upper);
            }

            // Generate samplesPerStratum points in this stratum
            for (size_t s = 0; s < samplesPerStratum; ++s) {
                std::vector<double> point(distributions.size());
                for (size_t dim = 0; dim < distributions.size(); ++dim) {
                    point[dim] = strataDists[dim](eng);
                }

                if (domain.contains(point)) {
                    sum += f(point);
                }
            }
        }

        return sum;
    };

    // Launch threads
    std::vector<std::future<double>> futures;
    futures.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        futures.push_back(std::async(std::launch::async, worker, i));
    }

    double totalSum = 0.0;
    for (auto &future: futures) {
        totalSum += future.get();
    }

    return totalSum * domain.getBoundedVolume() / static_cast<double>(numPoints);
}
