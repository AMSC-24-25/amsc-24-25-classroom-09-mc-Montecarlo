#ifndef HYPERSPHERE_H
#define HYPERSPHERE_H

#include "integration_domain.h"

#include <cmath>
#include <vector>
#include <utility>

// A hypersphere bounded by a hyper-rectangle
class Hypersphere : public IntegrationDomain {
    size_t dimensions;
    double radius;
    std::vector<std::pair<double, double>> bounds;

public:
    // Constructor that sets the dimensions and radius of the hypersphere, and initializes the bounding hyper-rectangle
    Hypersphere(size_t dims, double r);

    // Checks if a given point is inside the hypersphere
    bool contains(const std::vector<double> &point) const override;

    // Returns the bounding hyper-rectangle for the hypersphere
    std::vector<std::pair<double, double>> getBounds() const override;

    // Calculates the volume of the bounding hyper-rectangle
    double getBoundedVolume() const override;
};

#endif // HYPERSPHERE_H
