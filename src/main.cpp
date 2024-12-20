#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <cmath> // for M_PI, sqrt

#include "integrators/monte_carlo_integrator.h"
#include "domain/hypersphere.h"
#include "domain/polygon2d.h"
#include "integrators/metropolis_hastings_integrator.h"

int main() {
    // Function to integrate over the circle: x^2 + y^2
    auto f_circle = [](const std::vector<double> &x) {
        return x[0] * x[0] + x[1] * x[1];
    };

    // A circle with radius 1 centered at (0, 0)
    Hypersphere sphere(2, 1.0);

    // Monte Carlo integrator for the circle
    MonteCarloIntegrator mcIntegratorCircle(sphere);
    MetropolisHastingsIntegrator mhIntegratorCircle(sphere);

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) {
        numThreads = 16;
    }
    std::cout << "Using " << numThreads << " threads.\n";

    // Various numbers of points to test
    const std::vector<size_t> numPointsValues = {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000, 1'000'000'000};

    // For the circle domain test
    std::cout << "Integrating f(x,y)=x^2+y^2 over the unit circle (radius=1):\n";
    std::cout << "Expected result (π/2): " << std::fixed << std::setprecision(6) << (M_PI / 2) << "\n\n";
    std::cout << std::setw(12) << "NumPoints"
            << std::setw(16) << "Time (µs)"
            << std::setw(18) << "Standard Result"
            << std::setw(18) << "Stratified Result"
            << std::setw(18) << "Hastings Result"
            << std::setw(20) << "Hastings Accept(%)" << "\n";
    std::cout << std::string(101, '-') << "\n";

    for (size_t numPoints: numPointsValues) {
        // Standard Monte Carlo integration on the circle
        auto start = std::chrono::high_resolution_clock::now();
        double resultStandard = mcIntegratorCircle.integrate(f_circle, numPoints, numThreads);
        auto end = std::chrono::high_resolution_clock::now();
        auto durationStandard = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Stratified Monte Carlo integration on the circle
        int32_t strataPerDim = 10;
        start = std::chrono::high_resolution_clock::now();
        double resultStratified = mcIntegratorCircle.integrateStratified(f_circle, numPoints, numThreads, strataPerDim);
        end = std::chrono::high_resolution_clock::now();
        auto durationStratified = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Metropolis-Hastings integration on the circle
        size_t numPointsMH = numPoints / 10;
        std::vector<double> initialPoint = {0.0, 0.0};
        start = std::chrono::high_resolution_clock::now();
        auto [resultHastings, acceptanceRate] =
                mhIntegratorCircle.integrateParallel(f_circle, numPointsMH, initialPoint, numThreads);
        end = std::chrono::high_resolution_clock::now();
        auto durationHastings = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << std::setw(12) << numPoints
                << std::setw(15) << durationStandard.count()
                << std::setw(18) << std::fixed << std::setprecision(6) << resultStandard
                << std::setw(18) << std::fixed << std::setprecision(6) << resultStratified
                << std::setw(18) << std::fixed << std::setprecision(6) << resultHastings
                << std::setw(20) << std::fixed << std::setprecision(2) << (acceptanceRate * 100.0)
                << "\n";
    }

    // Now integrate over the (1,1,1) triangle domain with f(x,y)=1
    // Define the vertices of an equilateral triangle of side length 1
    // A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    std::vector<std::pair<double, double>> triangleVertices = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.5, std::sqrt(3.0) / 2.0}
    };

    Polygon2D triangle(triangleVertices);

    // Monte Carlo integrators for the triangle
    MonteCarloIntegrator mcIntegratorTriangle(triangle);
    MetropolisHastingsIntegrator mhIntegratorTriangle(triangle);

    // The function to integrate is f(x,y)=1
    auto f_identity = [](const std::vector<double> &) {
        return 1.0;
    };

    // The expected result is the area of the triangle = sqrt(3)/4
    double expectedTriangleArea = std::sqrt(3.0) / 4.0;

    std::cout << "\nIntegrating f(x,y)=1 over the equilateral triangle (1,1,1):\n";
    std::cout << "Expected area: " << std::fixed << std::setprecision(6) << expectedTriangleArea << "\n\n";
    std::cout << std::setw(12) << "NumPoints"
            << std::setw(16) << "Time (µs)"
            << std::setw(18) << "Standard Result"
            << std::setw(18) << "Stratified Result"
            << std::setw(18) << "Hastings Result"
            << std::setw(20) << "Hastings Accept(%)" << "\n";
    std::cout << std::string(101, '-') << "\n";

    for (size_t numPoints: numPointsValues) {
        // Standard Monte Carlo integration on the triangle
        auto start = std::chrono::high_resolution_clock::now();
        double resultStandard = mcIntegratorTriangle.integrate(f_identity, numPoints, numThreads);
        auto end = std::chrono::high_resolution_clock::now();
        auto durationStandard = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Stratified Monte Carlo integration on the triangle
        int32_t strataPerDim = 10;
        start = std::chrono::high_resolution_clock::now();
        double resultStratified = mcIntegratorTriangle.integrateStratified(
            f_identity, numPoints, numThreads, strataPerDim
        );
        end = std::chrono::high_resolution_clock::now();
        auto durationStratified = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Metropolis-Hastings integration on the triangle
        size_t numPointsMH = numPoints / 10;
        std::vector<double> initialPoint = {0.5, std::sqrt(3.0) / 6.0};
        start = std::chrono::high_resolution_clock::now();
        auto [resultHastings, acceptanceRate] =
                mhIntegratorTriangle.integrateParallel(f_identity, numPointsMH, initialPoint, numThreads);
        end = std::chrono::high_resolution_clock::now();
        auto durationHastings = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << std::setw(12) << numPoints
                << std::setw(15) << durationStandard.count()
                << std::setw(18) << std::fixed << std::setprecision(6) << resultStandard
                << std::setw(18) << std::fixed << std::setprecision(6) << resultStratified
                << std::setw(18) << std::fixed << std::setprecision(6) << resultHastings
                << std::setw(20) << std::fixed << std::setprecision(2) << (acceptanceRate * 100.0)
                << "\n";
    }

    return 0;
}
