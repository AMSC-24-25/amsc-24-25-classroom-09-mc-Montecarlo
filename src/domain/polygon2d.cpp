#include "polygon2d.h"

#include <cmath>
#include <stdexcept>

Polygon2D::Polygon2D(const std::vector<std::pair<double, double>> &verts) : vertices{verts} {}

bool Polygon2D::contains(const std::vector<double> &point) const {
    // Use ray-casting test to detect if the point belongs to the polygon
    if (point.size() != 2) {
        throw std::invalid_argument("Point must be 2D for Polygon2D");
    }

    double x = point[0], y = point[1];
    bool inside = false;
    size_t n = vertices.size();

    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        double xi = vertices[i].first, yi = vertices[i].second; // Current vertex (xi, yi)
        double xj = vertices[j].first, yj = vertices[j].second; // Previous vertex (xj, yj)

        // Check if the point is exactly on a vertex
        if ((x == xi && y == yi) || (x == xj && y == yj)) {
            return true; // Point is on a vertex
        }

        // Check if the point is exactly on the edge
        if ((yi > y) != (yj > y)) {
            double intersectionX = (xj - xi) * (y - yi) / (yj - yi) + xi;
            if (x == intersectionX) {
                return true; // Point is on the edge
            }
            if (x < intersectionX) {
                inside = !inside;
            }
        }

        // Check if the point is exactly on a horizontal edge
        if ((y == yi && y == yj) && (x >= std::min(xi, xj) && x <= std::max(xi, xj))) {
            return true; // Point is on the horizontal edge
        }

        // Check if the point is exactly on a vertical edge
        if ((x == xi && x == xj) && (y >= std::min(yi, yj) && y <= std::max(yi, yj))) {
            return true; // Point is on the vertical edge
        }
    }

    return inside;
}

std::vector<std::pair<double, double>> Polygon2D::getBounds() const {
    // Find the limits (minimum and maximum) for the x and y coordinates
    double xmin = vertices[0].first, xmax = vertices[0].first;
    double ymin = vertices[0].second, ymax = vertices[0].second;

    // Iterate over the vertices and update the minimum and maximum values for x and y
    for (const auto &[vx, vy]: vertices) {
        xmin = std::min(xmin, vx);
        xmax = std::max(xmax, vx);
        ymin = std::min(ymin, vy);
        ymax = std::max(ymax, vy);
    }

    return {{xmin, xmax}, {ymin, ymax}};
}

double Polygon2D::getBoundedVolume() const {
    auto bounds = getBounds();
    double width = bounds[0].second - bounds[0].first; // xmax - xmin
    double height = bounds[1].second - bounds[1].first; // ymax - ymin
    return width * height;
}
