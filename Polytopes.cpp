#include <Eigen/Dense>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <boost/random/sobol.hpp>

class IntegrationDomain {
public:
    virtual bool contains(const Eigen::VectorXd &point) const = 0;
    virtual std::pair<Eigen::VectorXd, Eigen::VectorXd> getBounds() const = 0;
    virtual double getVolume() const = 0;
    virtual ~IntegrationDomain() = default;
};

class Polytope : public IntegrationDomain {
    Eigen::MatrixXd A;
    Eigen::VectorXd b;

public:
    Polytope(const Eigen::MatrixXd &A_, const Eigen::VectorXd &b_)
        : A(A_), b(b_) {
        if (A.rows() != b.size()) {
            throw std::invalid_argument("Matrix A and vector b dimensions mismatch");
        }
    }

    bool contains(const Eigen::VectorXd &point) const override {
        return ((A * point).array() <= b.array()).all();
    }

    std::pair<Eigen::VectorXd, Eigen::VectorXd> getBounds() const override {
        Eigen::VectorXd lower = Eigen::VectorXd::Constant(A.cols(), -1.0);
        Eigen::VectorXd upper = Eigen::VectorXd::Constant(A.cols(), 1.0);
        return {lower, upper};
    }

    double getVolume() const override {
        throw std::logic_error("Volume computation not implemented.");
    }
};


class MonteCarloIntegrator {
    const IntegrationDomain &domain;

public:
    explicit MonteCarloIntegrator(const IntegrationDomain &d) : domain(d) {}

    double integrateSobol(
        const std::function<double(const Eigen::VectorXd &)> &f,
        size_t numPoints,
        size_t numThreads) {
        auto [lower, upper] = domain.getBounds();
        size_t dimensions = lower.size();

        double sum = 0.0;
        double sumSquared = 0.0;

        std::mutex mutex;

        auto worker = [&](size_t start, size_t end) {
            boost::random::sobol sobolGenerator(dimensions);
            Eigen::VectorXd sobolPoint(dimensions);
            Eigen::VectorXd scaledPoint(dimensions);

            double localSum = 0.0;
            double localSumSquared = 0.0;

            for (size_t i = start; i < end; ++i) {
                // Generate Sobol point
                for (size_t d = 0; d < dimensions; ++d) {
                    sobolPoint[d] = static_cast<double>(sobolGenerator()) /
                                    std::numeric_limits<boost::random::sobol::result_type>::max();
                }

                // Scale Sobol point
                scaledPoint = lower + sobolPoint.cwiseProduct(upper - lower); // This single-line operation is vectorized and eliminates temporary variables 
            

                // Evaluate function if within the domain
                if (domain.contains(scaledPoint)) {
                    double value = f(scaledPoint);
                    localSum += value;
                    localSumSquared += value * value;
                }
            }

            std::lock_guard<std::mutex> lock(mutex);
            sum += localSum;
            sumSquared += localSumSquared;
        };

        size_t pointsPerThread = numPoints / numThreads;
        std::vector<std::thread> threads;

        for (size_t i = 0; i < numThreads; ++i) {
            size_t start = i * pointsPerThread;
            size_t end = (i + 1 == numThreads) ? numPoints : start + pointsPerThread;
            threads.emplace_back(worker, start, end);
        }

        for (auto &t : threads) {
            t.join();
        }

        double mean = sum / numPoints;
        double variance = (sumSquared / numPoints) - (mean * mean);

        std::cout << "Variance: " << variance << std::endl;
        return mean;
    }
};

int main() {
    Eigen::MatrixXd A(6, 3);
    A << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;

    Eigen::VectorXd b(6);
    b << 1, 1, 1, 1, 1, 1;

    Polytope cube(A, b); // Instantiating Polytope
    MonteCarloIntegrator integrator(cube);

    auto f = [](const Eigen::VectorXd &x) { return x.squaredNorm(); };

    size_t numPoints = 1'000'000;
    size_t numThreads = std::thread::hardware_concurrency();

    std::cout << "Number of threads: " << numThreads << std::endl;

    double result = integrator.integrateSobol(f, numPoints, numThreads);
    std::cout << "Monte Carlo Sobol integration result: " << result << std::endl;

    return 0;
}

