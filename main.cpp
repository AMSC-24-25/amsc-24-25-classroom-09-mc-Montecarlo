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
    // Takes a point (represented as a coordinate vector), and checks whether that point belongs to the integration region. Returns a boolean value.
    virtual bool contains(const std::vector<double> &point) const = 0;

    // Returns the bounds of the integration region.
    virtual std::vector<std::pair<double, double>> getBounds() const = 0;

    // Calculates the volume of the integration domain bounds.
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
        // The volume of a hyper-rectangle (bounds of a sphere) in n dimensions is given by (2×radius)^n
        return std::pow(2 * radius, dimensions);
    }
};

class Polygon2D : public IntegrationDomain {
    std::vector<std::pair<double, double>> vertices;

public:
    explicit Polygon2D(const std::vector<std::pair<double, double>> &verts) : vertices{verts} {}

    bool contains(const std::vector<double> &point) const override {
        // Use ray-casting test to detect if the point belongs to the polygon
        if (point.size() != 2) {
            throw std::invalid_argument("Point must be 2D for Polygon2D");
        }

        double x = point[0], y = point[1];
        bool inside = false;
        size_t n = vertices.size();

        // Count how many times the horizontal ray (to the right of the point) intersects the sides of the polygon.
        // If the number of intersections is odd, the point is inside the polygon.
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            double xi = vertices[i].first, yi = vertices[i].second; // Current vertex (xi, yi)
            double xj = vertices[j].first, yj = vertices[j].second; // Previous vertex (xj, yj)

            bool intersectsVertically = (yi > y) != (yj > y);
            double intersectionX = (xj - xi) * (y - yi) / (yj - yi) + xi;
            if (intersectsVertically && x < intersectionX) {
                inside = !inside;
            }
        }

        return inside;
    }

    std::vector<std::pair<double, double>> getBounds() const override {
        // Find the limits (minimum and maximum) for the x and y coordinates.
        double xmin = vertices[0].first, xmax = vertices[0].first;
        double ymin = vertices[0].second, ymax = vertices[0].second;

        // Iterate over the vertices and update the minimum and maximum values for x and y.
        for (const auto &[x, y]: vertices) {
            xmin = std::min(xmin, x);
            xmax = std::max(xmax, x);
            ymin = std::min(ymin, y);
            ymax = std::max(ymax, y);
        }

        return {{xmin, xmax}, {ymin, ymax}};
    }

    double getVolume() const override {
        double area = 0.0;
        size_t n = vertices.size();

        // The area of a polygon can be calculated using the trapezoid formula.
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            area += (vertices[j].first * vertices[i].second - vertices[i].first * vertices[j].second);
        }

        return std::abs(area) / 2.0;
    }
};

class MonteCarloIntegrator {
    // Our domain of integration.
    const IntegrationDomain &domain;

    // Each distribution generates random numbers in a specific range, different for each dimension.
    std::vector<std::uniform_real_distribution<double>> distributions;

    // Random engines, mt19937 shows the best performance/quality balance.
    std::vector<std::mt19937> engines;

    // Initializes pseudo-random engines with seeds from std::seed_seq
    // std::seed_seq is initialized with non-deterministic random values from std::random_device.
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

    // Returns an n-dimensional point with pseudo-random independent coordinates.
    std::vector<double> generatePoint(std::mt19937 &eng) {
        std::vector<double> point(distributions.size());
        for (size_t i = 0; i < distributions.size(); ++i) {
            point[i] = distributions[i](eng);
        }
        return point;
    }

public:
    explicit MonteCarloIntegrator(const IntegrationDomain &d) : domain(d) {
        // Create a random-value uniform distribution in [min, max] for each dimension using the domain bounds.
        auto bounds = domain.getBounds();
        for (const auto &[min, max]: bounds) {
            distributions.emplace_back(min, max);
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

        // Retrieve partial results from each worker thread and combines them into a single total result.
        double totalSum = 0.0;
        for (auto &future: futures) {
            totalSum += future.get();
        }

        return domain.getVolume() * totalSum / static_cast<double>(numPoints);
    }

    // Stratified sampling
    std::vector<double> generateStratifiedPoint(std::mt19937 &eng, int32_t strataPerDimension) const {
        std::vector<double> point(distributions.size());

        for (size_t dim = 0; dim < distributions.size(); ++dim) {
            // Generate a random strata index.
            std::uniform_int_distribution<int32_t> strataDist(0, strataPerDimension - 1);
            int32_t strataIndex = strataDist(eng);

            // Calculate the layer boundaries.
            double step = (distributions[dim].b() - distributions[dim].a()) / strataPerDimension;
            double lower = distributions[dim].a() + strataIndex * step;
            double upper = lower + step;

            // Generate a random point within this layer.
            std::uniform_real_distribution<double> strataDistReal(lower, upper);
            point[dim] = strataDistReal(eng);
        }

        return point;
    }

    // New: method for stratified integration
    /*f: is the function we want to integrate. It takes a vector of coordinates (representing a point in the domain) and returns a value.
    numPoints: is the total number of points that will be generated to calculate the integral.
    numThreads: is the number of threads that will be used to parallelize the calculation.
    strataPerDimension: This is the number of "strata" (sub-domains) that will divide each dimension of the domain*/
    double integrateStratified(
        const std::function<double(const std::vector<double> &)> &f,
        size_t numPoints,
        size_t numThreads,
        int32_t strataPerDimension
    ) {
        initializeEngines(numThreads);
        size_t pointsPerThread = numPoints / numThreads;

        // Each thread works on a portion of the stitches
        auto worker = [this, &f, pointsPerThread, strataPerDimension](size_t myIndex) {
            double sum = 0.0;

            for (size_t i = 0; i < pointsPerThread; ++i) {
                // Generates a layered point
                auto point = generateStratifiedPoint(engines[myIndex], strataPerDimension);

                // If the point is in the domain, evaluate the function f and add the value to our partial result.
                if (domain.contains(point)) {
                    sum += f(point);
                }
            }

            return sum;
        };


        // Creating and launching threads
        std::vector<std::future<double>> futures;
        futures.reserve(numThreads);
        for (size_t i = 0; i < numThreads; ++i) {
            futures.push_back(
                std::async(std::launch::async, worker, i)
            );
        }

        // Retrieve partial results from each worker thread and combines them into a single total result.
        double totalSum = 0.0;
        for (auto &future: futures) {
            totalSum += future.get();
        }

        return domain.getVolume() * totalSum / static_cast<double>(numPoints);
    }
};

int main() {
    // Function to integrate: x^2 + y^2
    auto f = [](const std::vector<double> &x) {
        return x[0] * x[0] + x[1] * x[1];
    };

    // A circle with radius 1 centered at (0, 0)
    Hypersphere sphere(2, 1.0);
    MonteCarloIntegrator integrator(sphere);

    // The various thread numbers for testing
    const std::vector<size_t> threadCounts = {1, 2, 4, 8, 16, 32, 64};
    // Number of random points generated for integration
    constexpr size_t numPoints = 10000000;

    std::cout << "Monte Carlo integration with " << numPoints << " points\n\n";
    std::cout << std::setw(8) << "Threads"
            << std::setw(15) << "Time (µs)"
            << std::setw(15) << "Standard Result  "
            << std::setw(15) << "Stratified Result  "
            << std::setw(15) << " Hastings Result  " << "\n";
    std::cout << std::string(68, '-') << "\n";

    // Run the integration with different numbers of threads and compare the results
    for (size_t threads: threadCounts) {
        // Integration with the standard method
        auto start = std::chrono::high_resolution_clock::now();
        double resultStandard = integrator.integrate(f, numPoints, threads);
        auto end = std::chrono::high_resolution_clock::now();
        auto durationStandard = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Integration with Stratified Sampling
        int32_t strataPerDimension = 10; // Number of layers per dimension
        start = std::chrono::high_resolution_clock::now();
        double resultStratified = integrator.integrateStratified(f, numPoints, threads, strataPerDimension);
        end = std::chrono::high_resolution_clock::now();
        auto durationStratified = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Results
        std::cout << std::setw(8) << threads
                << std::setw(15) << durationStandard.count()
                << std::setw(15) << std::fixed << std::setprecision(6) << resultStandard
                << std::setw(15) << std::fixed << std::setprecision(6) << resultStratified << "\n";
    }

    return 0;
}
