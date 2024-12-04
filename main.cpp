#include <random>
#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <thread>
#include <future>
#include <chrono>
#include <iomanip>

// An interface for different domains
class IntegrationDomain {
public:
    virtual bool contains(const std::vector<double> &point) const = 0;
    virtual std::vector<std::pair<double, double>> getBounds() const = 0;
    virtual double getVolume() const = 0;
    virtual ~IntegrationDomain() = default;
};

// A hypersphere bounded by a hyper-rectangle
class Hypersphere : public IntegrationDomain {
    size_t dimensions;
    double radius;
    std::vector<std::pair<double, double>> bounds;

public:
    Hypersphere(size_t dims, double r) : dimensions{dims}, radius{r} {
        bounds.resize(dimensions, {-radius, radius});
    }

    bool contains(const std::vector<double> &point) const override {
        double sumSquares = 0.0;
        for (const auto &x: point) {
            sumSquares += x * x;
        }
        return sumSquares <= radius * radius;
    }

    std::vector<std::pair<double, double>> getBounds() const override {
        return bounds;
    }

    double getVolume() const override {
        return std::pow(2 * radius, dimensions);
    }
};

// TODO: `More complex domains: a general polygon (in 2D) or even polytopes in nD.`

class MonteCarloIntegrator {
    std::vector<std::uniform_real_distribution<double>> distributions;
    std::vector<std::mt19937> engines; // TODO: use different types of engines?
    const IntegrationDomain &domain;

    // Initializes pseudo-random engines with seeds from std::seed_seq
    // std::seed_seq is initialized with non-deterministic random values from std::random_device
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

    // Returns an n-dimensional point with pseudo-random independent coordinates
    std::vector<double> generatePoint(std::mt19937 &eng) {
        std::vector<double> point(distributions.size());
        for (size_t i = 0; i < distributions.size(); ++i) {
            point[i] = distributions[i](eng);
        }
        return point;
    }

public:
    explicit MonteCarloIntegrator(const IntegrationDomain &d) : domain(d) {
        auto bounds = domain.getBounds();
        for (const auto &bound: bounds) {
            distributions.emplace_back(bound.first, bound.second);
        }
    }

    // Integrates real-valued function `f` over the domain specified in the constructor
    double integrate(
        const std::function<double(const std::vector<double> &)> &f,
        size_t numPoints,
        size_t numThreads
    ) {
        initializeEngines(numThreads);
        size_t pointsPerThread = numPoints / numThreads;

        // This lambda is invocated for each worker thread
        auto worker = [this, &f, pointsPerThread](size_t myIndex) {
            double sum = 0.0;

            for (size_t i = 0; i < pointsPerThread; ++i) {
                auto point = generatePoint(engines[myIndex]);
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
            futures.push_back(std::async(
                std::launch::async,
                worker,
                i
            ));
        }

        // Combine results from each worker thread
        double totalSum = 0.0;
        for (auto &future: futures) {
            totalSum += future.get();
        }

        return domain.getVolume() * totalSum / static_cast<double>(numPoints);
    }
};

// TODO: `Use different type of sampling: stratified sampling, importance sampling (Metropolis-Hastings).`

int main() {
    // Function to integrate: x^2 + y^2
    auto f = [](const std::vector<double> &x) {
        return x[0] * x[0] + x[1] * x[1];
    };

    // A circle with radius 1 with center in (0, 0)
    Hypersphere sphere(2, 1.0);
    MonteCarloIntegrator integrator(sphere);

    const std::vector<size_t> threadCounts = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    constexpr size_t numPoints = 10000000; // 10^7

    std::cout << "Integration with " << numPoints << " points\n\n";
    std::cout << std::setw(8) << "Threads"
            << std::setw(15) << "Time (µs)"
            << std::setw(15) << "Result" << "\n";
    std::cout << std::string(38, '-') << "\n";

    // Must show an increase of speed until a certain point (which is optimal #threads on your machine)
    for (size_t threads: threadCounts) {
        auto start = std::chrono::high_resolution_clock::now();
        double result = integrator.integrate(f, numPoints, threads);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Must be close to pi/2
        std::cout << std::setw(8) << threads
                << std::setw(15) << duration.count()
                << std::setw(15) << std::fixed << std::setprecision(6) << result << "\n";
    }

    // TODO: find best #threads and iterate over increasing #points to show improvement in precision?

    return 0;
}
