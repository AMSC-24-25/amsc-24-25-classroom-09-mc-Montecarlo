#ifndef ABSTRACT_INTEGRATOR_H
#define ABSTRACT_INTEGRATOR_H

#include <random>
#include <vector>

#include "domain/integration_domain.h"

class AbstractIntegrator {
protected:
    // Our domain of integration
    const IntegrationDomain &domain;

    // Random engines shared by all integrator implementations
    std::vector<std::mt19937> engines;

    // Initializes pseudo-random engines with seeds from std::seed_seq
    void initializeEngines(size_t numThreads) {
        std::random_device rd;
        std::vector<std::uint32_t> entropy;
        entropy.reserve(numThreads);

        for (size_t i = 0; i < numThreads; ++i) {
            entropy.push_back(rd());
        }

        std::seed_seq seq(entropy.begin(), entropy.end());
        std::vector<std::uint32_t> seeds(numThreads);
        seq.generate(seeds.begin(), seeds.end());

        engines.clear();
        for (auto seed: seeds) {
            engines.emplace_back(seed);
        }
    }

public:
    explicit AbstractIntegrator(const IntegrationDomain &d) : domain(d) {}

    virtual ~AbstractIntegrator() = default;
};

#endif // ABSTRACT_INTEGRATOR_H
