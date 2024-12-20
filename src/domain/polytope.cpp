#include "polytope.h"

#include <thread>
#include <future>
#include <random>
#include <stdexcept>

Polytope::Polytope(const Eigen::MatrixXd &A_, const Eigen::VectorXd &b_, const std::pair<Eigen::VectorXd, Eigen::VectorXd> &explicitBounds)
    : A(A_), b(b_), bounds(explicitBounds)
{
    if (A.rows() != b.size()) {
        throw std::invalid_argument("A and b's dimensions aren't compatible.");
    }

    if (bounds.first.size() != A.cols() || bounds.second.size() != A.cols()) {
        throw std::invalid_argument("Bounds must match the dimensionality of the polytope.");
    }
}

bool Polytope::contains(const std::vector<double> &point) const {
    if (point.size() != static_cast<size_t>(A.cols())) {
        throw std::invalid_argument("Point dimension does not match polytope dimension.");
    }

    Eigen::VectorXd pointVec(A.cols());
    for (int i = 0; i < A.cols(); ++i) {
        pointVec[i] = point[i];
    }

    // Check A * pointVec ≤ b
    return ((A * pointVec).array() <= b.array()).all();
}

std::vector<std::pair<double, double>> Polytope::getBounds() const {
    std::vector<std::pair<double, double>> box(bounds.first.size());
    for (int i = 0; i < bounds.first.size(); ++i) {
        box[i] = {bounds.first[i], bounds.second[i]};
    }
    return box;
}

double Polytope::getBoundedVolume() const {
    // Return the volume of the bounding box
    double volume = 1.0;
    for (int i = 0; i < bounds.first.size(); ++i) {
        volume *= (bounds.second[i] - bounds.first[i]);
    }
    return volume;
}
