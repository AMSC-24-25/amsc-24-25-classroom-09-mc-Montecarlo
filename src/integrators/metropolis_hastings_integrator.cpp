#include "metropolis_hastings_integrator.h"

MetropolisHastingsIntegrator::MetropolisHastingsIntegrator(const IntegrationDomain &d)
    : domain(d), proposalDist(0.0, 0.01) {}

// Initializes random engines with unique seeds for each chain
std::vector<std::mt19937> MetropolisHastingsIntegrator::initializeEngines(size_t numChains) {
    std::random_device rd;
    std::vector<std::mt19937> engines(numChains);

    // Initialize each engine with a unique seed.
    for (size_t i = 0; i < numChains; ++i) {
        engines[i] = std::mt19937(rd());
    }

    return engines;
}

// Perform a single Metropolis-Hastings chain
std::pair<double, int32_t> MetropolisHastingsIntegrator::integrateSingleChain(
    const std::function<double(const std::vector<double> &)> &f,
    size_t numPoints,
    const std::vector<double> &initialPoint,
    std::mt19937 &engine
) {
    const size_t dimensions = initialPoint.size();
    std::vector<double> currentPoint = initialPoint;
    double currentFValue = f(currentPoint);
    double sumF = 0.0;
    int32_t accepted = 0;

    std::vector<double> candidatePoint(dimensions);
    std::uniform_real_distribution<> unifDist(0.0, 1.0);

    // Main sampling loop
    for (size_t i = 0; i < numPoints; ++i) {
        // Generate candidate point by adding Gaussian noise
        for (size_t d = 0; d < dimensions; ++d) {
            candidatePoint[d] = currentPoint[d] + proposalDist(engine);
        }

        // Check if the candidate point is within the bounded region
        if (domain.boundContains(candidatePoint)) {
            double candidateFValue = f(candidatePoint);

            // Compute the Metropolis-Hastings acceptance ratio.
            // Use max to avoid numerical issues with very small values
            double acceptanceRatio = std::max(1e-10, candidateFValue / std::max(1e-10, currentFValue));

            if (unifDist(engine) <= acceptanceRatio) {
                currentPoint = candidatePoint;
                currentFValue = candidateFValue;
                ++accepted;

                // Only add to sum if point is in actual domain
                if (domain.contains(currentPoint)) {
                    sumF += currentFValue;
                }
            }
        }
    }

    // Return the partial integral result scaled by the domain's volume
    return {sumF * domain.getBoundedVolume() / accepted, accepted};
}

// Perform parallel integration with multiple chains
std::pair<double, double> MetropolisHastingsIntegrator::integrateParallel(
    const std::function<double(const std::vector<double> &)> &f,
    size_t numPoints,
    const std::vector<double> &initialPoint,
    size_t numChains
) {
    // Initialize random engines for each chain
    auto engines = initializeEngines(numChains);
    size_t pointsPerChain = numPoints / numChains;

    // Launch parallel chains using std::async
    std::vector<std::future<std::pair<double, int32_t>>> futures;
    futures.reserve(numChains);
    for (size_t i = 0; i < numChains; ++i) {
        futures.push_back(std::async(
            std::launch::async,
            [this, &f, pointsPerChain, &initialPoint, &engine = engines[i]]() {
                return this->integrateSingleChain(f, pointsPerChain, initialPoint, engine);
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
