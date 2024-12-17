#ifndef POLYTOPE_H
#define POLYTOPE_H

#include "integration_domain.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <utility>
#include <vector>

class Polytope : public IntegrationDomain {
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::pair<Eigen::VectorXd, Eigen::VectorXd> bounds;

public:
    Polytope(const Eigen::MatrixXd &A_, const Eigen::VectorXd &b_, const std::pair<Eigen::VectorXd, Eigen::VectorXd> &explicitBounds);

    bool contains(const std::vector<double> &point) const override;
    std::vector<std::pair<double, double>> getBounds() const override;
    double getBoundedVolume() const override;
};

#endif // POLYTOPE_H
