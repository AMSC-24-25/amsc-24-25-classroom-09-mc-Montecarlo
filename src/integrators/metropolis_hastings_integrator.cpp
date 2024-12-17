#include "metropolis_hastings_integrator.h"

MetropolisHastingsIntegrator::MetropolisHastingsIntegrator(const IntegrationDomain &d)
    : domain(d), proposalDist(0.0, 1.0) {} // Gaussian proposal (mean=0, stddev=1)

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
double MetropolisHastingsIntegrator::integrateSingleChain(
    const std::function<double(const std::vector<double> &)> &f,
    size_t numPoints,
    const std::vector<double> &initialPoint,
    std::mt19937 &engine,
    size_t &accepted
) {
    // Number of dimensions
    size_t dimensions = initialPoint.size();
    std::vector<double> currentPoint = initialPoint;
    double currentFValue = f(currentPoint);
    double sumF = 0.0;
    accepted = 0;

    for (size_t i = 0; i < numPoints; ++i) {
        // Generate a candidate point by adding Gaussian noise
        std::vector<double> candidatePoint(dimensions);
        for (size_t d = 0; d < dimensions; ++d) {
            candidatePoint[d] = currentPoint[d] + proposalDist(engine);
        }

        // Check if the candidate point is within the domain.
        if (domain.contains(candidatePoint)) {
            double candidateFValue = f(candidatePoint);

            // Compute the Metropolis-Hastings acceptance ratio.
            double acceptanceRatio = candidateFValue / currentFValue;

            // Accept or reject the candidate point.
            if (std::uniform_real_distribution<>(0.0, 1.0)(engine) <= acceptanceRatio) {
                currentPoint = candidatePoint;
                currentFValue = candidateFValue;
                ++accepted;
            }
        }

        // Add the current function value to the cumulative sum.
        sumF += currentFValue;
    }

    // Return the partial integral result scaled by the domain's volume.
    return domain.getBoundedVolume() * sumF / static_cast<double>(numPoints);
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

    // Points per chain
    size_t pointsPerChain = numPoints / numChains;

    // Launch parallel chains using std::async
    std::vector<std::future<std::pair<double, size_t>>> futures;
    futures.reserve(numChains);
    for (size_t i = 0; i < numChains; ++i) {
        futures.push_back(std::async(
            std::launch::async,
            [this, &f, pointsPerChain, &initialPoint, &engine = engines[i]]() {
                size_t accepted = 0;
                double result = this->integrateSingleChain(f, pointsPerChain, initialPoint, engine, accepted);
                return std::make_pair(result, accepted);
            }
        ));
    }

    // Collect results
    double totalResult = 0.0;
    size_t totalAccepted = 0;
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
