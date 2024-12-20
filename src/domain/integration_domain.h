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

    // Returns true if the point belongs to the bounded region
    bool boundContains(const std::vector<double> &point) const {
        auto bounds = getBounds();
        bool ok = true;
        for (int i = 0; i < point.size(); i++) {
            if (point[i] < bounds[i].first || point[i] > bounds[i].second) {
                ok = false;
                break;
            }
        }
        return ok;
    }

    virtual ~IntegrationDomain() = default;
};

#endif // INTEGRATION_DOMAIN_H
