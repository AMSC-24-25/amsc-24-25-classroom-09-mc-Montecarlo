#ifndef POLYGON2D_H
#define POLYGON2D_H

#include "integration_domain.h"

#include <vector>
#include <utility>

// A polygonal domain in 2D
class Polygon2D : public IntegrationDomain {
    std::vector<std::pair<double, double>> vertices;

public:
    // Constructor that takes a set of vertices defining the polygon
    explicit Polygon2D(const std::vector<std::pair<double, double>> &verts);

    // Uses the ray-casting algorithm to determine if a point is inside the polygon
    bool contains(const std::vector<double> &point) const override;

    // Returns the bounding box of the polygon
    std::vector<std::pair<double, double>> getBounds() const override;

    // Returns the area of the polygon (considered the "volume" in 2D)
    double getBoundedVolume() const override;
};

#endif // POLYGON2D_H
