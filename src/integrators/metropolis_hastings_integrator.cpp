#include "metropolis_hastings_integrator.h"

MetropolisHastingsIntegrator::MetropolisHastingsIntegrator(const IntegrationDomain &d, double stddev)
    : AbstractIntegrator(d), proposalDist(0.0, stddev) {}

// Perform a single Metropolis-Hastings chain
std::pair<double, int32_t> MetropolisHastingsIntegrator::integrateSingleChain(
    const std::function<double(const std::vector<double> &)> &f, // Function to get expectation of
    const std::function<double(const std::vector<double> &)> &p, // Target distribution
    size_t numPoints,
    const std::vector<double> &initialPoint,
    std::mt19937 &engine
) {
    const size_t dimensions = initialPoint.size();
    std::vector<double> currentPoint = initialPoint;
    double currentP = p(currentPoint);

    double sumF = 0.0;
    int32_t accepted = 0;
    int32_t samples = 0;

    std::vector<double> candidatePoint(dimensions);
    std::uniform_real_distribution<> unifDist(0.0, 1.0);

    // Main sampling loop
    for (size_t i = 0; i < numPoints; ++i) {
        // Generate candidate point by adding Gaussian noise
        for (size_t d = 0; d < dimensions; ++d) {
            candidatePoint[d] = currentPoint[d] + proposalDist(engine);
        }

        // Check if the candidate point is within the bounded region
        if (domain.contains(candidatePoint)) {
            double candidateP = p(candidatePoint);

            // MH acceptance ratio for sampling from p(x)
            double acceptanceRatio = candidateP / currentP;

            if (unifDist(engine) <= acceptanceRatio) {
                currentPoint = candidatePoint;
                currentP = candidateP;
                ++accepted;
            }

            sumF += f(currentPoint);
            ++samples;
        }
    }

    // Return the expectation E[f(x)] = average of f(x) over samples from p(x)
    return {sumF / samples, accepted};
}

// Perform parallel integration with multiple chains
std::pair<double, double> MetropolisHastingsIntegrator::integrateParallel(
    const std::function<double(const std::vector<double> &)> &f,
    const std::function<double(const std::vector<double> &)> &p,
    size_t numPoints,
    const std::vector<double> &initialPoint,
    size_t numChains
) {
    // Initialize random engines for each chain
    initializeEngines(numChains);
    size_t pointsPerChain = numPoints / numChains;

    // Launch parallel chains using std::async
    std::vector<std::future<std::pair<double, int32_t>>> futures;
    futures.reserve(numChains);
    for (size_t i = 0; i < numChains; ++i) {
        futures.push_back(std::async(
            std::launch::async,
            [this, &f, &p, pointsPerChain, &initialPoint, &engine = engines[i]]() {
                return this->integrateSingleChain(f, p, pointsPerChain, initialPoint, engine);
            }
        ));
    }

    // Collect results
    double totalResult = 0.0;
    int32_t totalAccepted = 0;
    for (auto &future: futures) {
        auto [result, accepted] = future.get();
        totalResult += result;
        totalAccepted += accepted;
    }

    // The overall acceptance rate
    double acceptanceRate = static_cast<double>(totalAccepted) / numPoints;

    // Return the combined result (average of chain results)
    return {totalResult / numChains, acceptanceRate};
}
