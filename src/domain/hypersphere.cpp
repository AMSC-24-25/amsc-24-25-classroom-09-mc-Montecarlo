#include "hypersphere.h"

#include <algorithm>

Hypersphere::Hypersphere(size_t dims, double r) : dimensions{dims}, radius{r} {
    // Initialize all bounds to [-r, r]
    bounds.resize(dimensions, {-radius, radius});
}

bool Hypersphere::contains(const std::vector<double> &point) const {
    double sumSquares = 0.0;
    for (const auto &x: point) {
        sumSquares += x * x;
    }
    return sumSquares <= radius * radius;
}

std::vector<std::pair<double, double>> Hypersphere::getBounds() const {
    return bounds;
}

double Hypersphere::getBoundedVolume() const {
    // The volume of a hyper-rectangle (bounds of a sphere) in n dimensions is given by (2×radius)^n
    return std::pow(2 * radius, dimensions);
}
