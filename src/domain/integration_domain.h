#ifndef INTEGRATION_DOMAIN_H
#define INTEGRATION_DOMAIN_H

#include <vector>
#include <utility>

// An interface for different domains
class IntegrationDomain {
public:
    // Returns the bounds of the integration region
    virtual std::vector<std::pair<double, double>> getBounds() const = 0;

    // Calculates the volume of the integration domain bounds
    virtual double getBoundedVolume() const = 0;

    // Takes a point (represented as a coordinate vector), and checks whether that point belongs to the integration region
    virtual bool contains(const std::vector<double> &point) const = 0;

    virtual ~IntegrationDomain() = default;
};

#endif // INTEGRATION_DOMAIN_H
