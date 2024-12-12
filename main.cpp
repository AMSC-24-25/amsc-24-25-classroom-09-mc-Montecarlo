#include <random>
#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <thread>
#include <future>
#include <chrono>
#include <iomanip>

// An interface for different domains
class IntegrationDomain {
public:
    virtual bool contains(const std::vector<double> &point) const = 0;        //takes a point (represented as a coordinate vector), and checks whether that point belongs to the integration region. Returns a boolean value
    virtual std::vector<std::pair<double, double>> getBounds() const = 0;     //getBounds returns the limits of the integration region.
    virtual double getVolume() const = 0;                                     //getVolume calculates the volume of the integration region  (can be an area, a volume or something else)
    virtual ~IntegrationDomain() = default;                                   //destroyer
};

// A hypersphere bounded by a hyper-rectangle
class Hypersphere : public IntegrationDomain {
    size_t dimensions;                                                        //dimensions of the hyper-sphere
    double radius;                                                            //the radius of the hyper-sphere
    std::vector<std::pair<double, double>> bounds;                            //lower and upper limit of each dimension of the hyper-sphere.


public:
    Hypersphere(size_t dims, double r) : dimensions{dims}, radius{r} {        //class constructor
        bounds.resize(dimensions, {-radius, radius});
    }

    bool contains(const std::vector<double> &point) const override {          //contains tests whether a point is contained within the hyper-sphere.
        double sumSquares = 0.0;
        for (const auto &x: point) {
            sumSquares += x * x;                                              //calculation of the squared distance from the center of the hyper-sphere.
        }
        return sumSquares <= radius * radius;
    }

    std::vector<std::pair<double, double>> getBounds() const override {      //returns the bounds vector
        return bounds;
    }

    double getVolume() const override {                                      //calculates and returns the volume of the hyper-sphere.
        return std::pow(2 * radius, dimensions);                             //volume of a hyper-rectum in n dimensions is given by (2×radius)^n
    }
};

// TODO: `More complex domains: a general polygon (in 2D) or even polytopes in nD.`
//->
class Polygon2D : public IntegrationDomain {
    std::vector<std::pair<double, double>> vertices;                            // list of polygon vertices

public:
    explicit Polygon2D(const std::vector<std::pair<double, double>> &verts) : vertices{verts} {}

    bool contains(const std::vector<double> &point) const override {            //The contains method uses ray-casting test
        if (point.size() != 2) {
            throw std::invalid_argument("Point must be 2D for Polygon2D");
        }

        double x = point[0], y = point[1];
        bool inside = false;
        size_t n = vertices.size();

        for (size_t i = 0, j = n - 1; i < n; j = i++) {                         //loop on the sides
            double xi = vertices[i].first, yi = vertices[i].second;             // Current vertex (xi, yi)
            double xj = vertices[j].first, yj = vertices[j].second;             // Previous vertex (xj, yj)
                                                                                           
            if (((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {   //Count how many times the ray intersects the sides of the polygon.
                inside = !inside;                                                          //If the number of intersections is odd, the point is inside the polygon
            }                                                                              //Otherwise, it's out.
        }
        //(yi > y) != (yj > y): This check verifies that the ray actually passes through the side of the polygon
        return inside;
    }

    std::vector<std::pair<double, double>> getBounds() const override {   //Find the limits (minimum and maximum) for the x and y coordinates:
        double xmin = vertices[0].first, xmax = vertices[0].first;
        double ymin = vertices[0].second, ymax = vertices[0].second;

        for (const auto &v : vertices) {                                  //Scroll through the vertices and update the minimum and maximum values ​​for x and y
            xmin = std::min(xmin, v.first);
            xmax = std::max(xmax, v.first);
            ymin = std::min(ymin, v.second);
            ymax = std::max(ymax, v.second);
        }

        return {{xmin, xmax}, {ymin, ymax}};
    }

    double getVolume() const override {
        double area = 0.0;
        size_t n = vertices.size();                                          //The area of ​​a polygon can be calculated using the trapezoid formula

        for (size_t i = 0, j = n - 1; i < n; j = i++) {                                     
            area += (vertices[j].first * vertices[i].second - vertices[i].first * vertices[j].second);
        }
        return std::abs(area) / 2.0;
    }
};

class MonteCarloIntegrator {
    std::vector<std::uniform_real_distribution<double>> distributions;                  //distributions: a vector of uniform distributions. Each distribution generates random numbers in a specific range, one for each domain size
    std::vector<std::mt19937> engines; // TODO: use different types of engines?         //vector of random number generators
    const IntegrationDomain &domain;                                                    //integration domain

    // Initializes pseudo-random engines with seeds from std::seed_seq
    // std::seed_seq is initialized with non-deterministic random values from std::random_device
    void initializeEngines(size_t numThreads) {
        std::random_device rd;                                                          //provides random values
        std::vector<std::uint32_t> entropy;                                             //entropy: accommodates these values ​​to create a seed sequence (std::seed_seq), 
        entropy.reserve(numThreads);                                                    //which ensures that each generator (std::mt19937) has a different seed.

        for (size_t i = 0; i < numThreads; ++i) {
            entropy.push_back(rd());
        }

        std::seed_seq seq(entropy.begin(), entropy.end());
        std::vector<std::uint32_t> seeds(numThreads);
        seq.generate(seeds.begin(), seeds.end());

        engines.clear();
        for (auto seed: seeds) {
            engines.emplace_back(seed);
        }
    }

    // Returns an n-dimensional point with pseudo-random independent coordinates
    std::vector<double> generatePoint(std::mt19937 &eng) {                              //Generates a random point in a multidimensional domain
        std::vector<double> point(distributions.size());
        for (size_t i = 0; i < distributions.size(); ++i) {
            point[i] = distributions[i](eng);
        }
        return point;
    }

 public:
    explicit MonteCarloIntegrator(const IntegrationDomain &d) : domain(d) {             //Gets the bounds of the domain and creates a uniform 
        auto bounds = domain.getBounds();                                               //distribution for each dimension, using the lower (bound.first) and upper (bound.second) boundaries.
        for (const auto &bound: bounds) {
            distributions.emplace_back(bound.first, bound.second);
        }
    }

    // Integrates real-valued function `f` over the domain specified in the constructor
    double integrate(                                                                       
        const std::function<double(const std::vector<double> &)> &f,
        size_t numPoints,
        size_t numThreads
    ) {
        initializeEngines(numThreads);                                                    ////Initialize the random generators for the required number of threads
        size_t pointsPerThread = numPoints / numThreads;                                  //Divides the total number of points numPoints to be generated equally among threads.

        // This lambda is invocated for each worker thread
        auto worker = [this, &f, pointsPerThread](size_t myIndex) {                       //represents the work of a single thread
            double sum = 0.0;

            for (size_t i = 0; i < pointsPerThread; ++i) {
                auto point = generatePoint(engines[myIndex]);
                if (domain.contains(point)) {                                             //For each point, check whether it is contained in the domain
                    sum += f(point);                                                      //If the point is valid, evaluate the function f at that point and add the value to the partial result.
                }
            }

            return sum;
        };

        std::vector<std::future<double>> futures;                                      

        // Start numThreads worker threads                                               //Creating threads
        futures.reserve(numThreads);
        for (size_t i = 0; i < numThreads; ++i) {
            futures.push_back(std::async(
                std::launch::async,
                worker,
                i
            ));
        }

        // Combine results from each worker thread                                      //Retrieves partial results from each thread and combines them into a single total result.
        double totalSum = 0.0;
        for (auto &future: futures) {
            totalSum += future.get();
        }

        return domain.getVolume() * totalSum / static_cast<double>(numPoints);
    }


    //Stratified sampling
    // New: method for layering-->method for stratified integration
     std::vector<double> generateStratifiedPoint(std::mt19937 &eng, size_t strataPerDimension) {   // strataPerDimensionis: is the number of layers for each domain dimension. 
     std::vector<double> point(distributions.size());                                              // std::mt19937 &eng: is the random number generator
                                                                                                   // point: it is a vector that will contain the coordinates of the layered point we are generating
        // for each size
       for (size_t dim = 0; dim < distributions.size(); ++dim) {
        //Generate a random number to choose the layer
        std::uniform_int_distribution<size_t> strataDist(0, strataPerDimension - 1);                //strataDist is a discrete uniform distribution that generates a random number between 0 and strataPerDimension - 1. 
        size_t stratum = strataDist(eng);                                                           //This number represents the selected stratum for the current dimension.
        //stratum is the index of the randomly chosen stratum in this dimension.

        //Calculate the layer boundaries
        double lower = distributions[dim].a() + stratum * (distributions[dim].b() - distributions[dim].a()) / strataPerDimension;
        double upper = lower + (distributions[dim].b() - distributions[dim].a()) / strataPerDimension;

        //Generate the random point within this layer
        std::uniform_real_distribution<double> strataDistReal(lower, upper);   
        point[dim] = strataDistReal(eng);                                                            //point[dim] assigns point[dim] a random value within the current layer of the dim dimension.
     }
     return point;
    }

    // New: method for stratified integration
    /*f: is the function we want to integrate. It takes a vector of coordinates (representing a point in the domain) and returns a value.
    numPoints: is the total number of points that will be generated to calculate the integral.
    numThreads: is the number of threads that will be used to parallelize the calculation.
    strataPerDimension: This is the number of "strata" (sub-domains) that will divide each dimension of the domain*/
    double integrateStratified(
    const std::function<double(const std::vector<double> &)> &f,
    size_t numPoints, size_t numThreads, size_t strataPerDimension) {
     initializeEngines(numThreads);
     size_t pointsPerThread = numPoints / numThreads;                         //Calculate the points for each thread

     // Each thread works on a portion of the stitches                               
     auto worker = [this, &f, pointsPerThread, strataPerDimension](size_t myIndex) {     //lambda worker (worker function for each thread)
        double sum = 0.0;
        for (size_t i = 0; i < pointsPerThread; ++i) {
            // Generates a layered point
            auto point = generateStratifiedPoint(engines[myIndex], strataPerDimension);          
            if (domain.contains(point)) {
                sum += f(point);               //Sum the function values
            }
            //Once the point has been generated, the program checks whether the point belongs to the domain (domain.contains(point)).
            //If yes, the function f is calculated for that point and the result is added.
        }
        return sum;
     };


     //Creating and launching threads
     std::vector<std::future<double>> futures;
     futures.reserve(numThreads);
     for (size_t i = 0; i < numThreads; ++i) {
         futures.push_back(std::async(std::launch::async, worker, i));
     }

     //Combine the results from all threads
     double totalSum = 0.0;
     for (auto &future : futures) {
        totalSum += future.get();
     }

     //Returns the result of the integration
     return domain.getVolume() * totalSum / static_cast<double>(numPoints);

    }

     // New method: Metropolis-Hastings sampling
        std::vector<double> generateHastingsPoint(std::mt19937 &eng, const std::function<double(const std::vector<double> &)> &f) {
        std::vector<double> point = generatePoint(eng);
        std::normal_distribution<double> proposalDist(0.0, 1.0);
        std::vector<double> proposedPoint = point;

        for (size_t i = 0; i < point.size(); ++i) {
            proposedPoint[i] += proposalDist(eng);
        }

        double currentValue = f(point);  // Function value for the current point
        double proposedValue = f(proposedPoint);  // Function value for the proposed point

        if (domain.contains(proposedPoint) && std::exp(proposedValue - currentValue) > std::uniform_real_distribution<double>(0.0, 1.0)(eng)) {
            point = proposedPoint;  // Accept the new point if it's better or accepted based on the criterion
        }

        return point;
            }


    // New method for integration using Hastings sampling
    double integrateHastings(
    const std::function<double(const std::vector<double> &)> &f,
    size_t numPoints, size_t numThreads
  ) {
    initializeEngines(numThreads);
    size_t pointsPerThread = numPoints / numThreads;

    auto worker = [this, &f, pointsPerThread](size_t myIndex) {
        double sum = 0.0;

        for (size_t i = 0; i < pointsPerThread; ++i) {
            // Pass the function f to generateHastingsPoint
            auto point = generateHastingsPoint(engines[myIndex], f);
            if (domain.contains(point)) {
                sum += f(point);
            }
        }
        return sum;
    };

    std::vector<std::future<double>> futures;
    futures.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        futures.push_back(std::async(std::launch::async, worker, i));
    }

    double totalSum = 0.0;
    for (auto &future : futures) {
        totalSum += future.get();
    }

    return domain.getVolume() * totalSum / static_cast<double>(numPoints);
    }
};


int main() {
   // Function to integrate: x^2 + y^2
    auto f = [](const std::vector<double> &x) {
        return x[0] * x[0] + x[1] * x[1]; 
    };

    // A circle with radius 1 centered at (0, 0)
    Hypersphere sphere(2, 1.0);  // Integration on a sphere of dimension 2 and radius 1
    MonteCarloIntegrator integrator(sphere);

    const std::vector<size_t> threadCounts = {1, 2, 4, 8, 16, 32, 64};  // The various thread numbers for testing
    constexpr size_t numPoints = 10000000;  // Number of random points generated for integration

    std::cout << "Monte Carlo integration with " << numPoints << " points\n\n";
    std::cout << std::setw(8) << "Threads"
              << std::setw(15) << "Time (µs)"
              << std::setw(15) << "Standard Result  "
              << std::setw(15) << "Stratified Result  "
              << std::setw(15) << " Hastings Result  " << "\n";  // Add column for Hastings
    std::cout << std::string(68, '-') << "\n";  // Adjust separator to match the new columns

    // Run the integration with different numbers of threads and compare the results
    for (size_t threads : threadCounts) {
        // Integration with the standard method
        auto start = std::chrono::high_resolution_clock::now();
        double resultStandard = integrator.integrate(f, numPoints, threads);
        auto end = std::chrono::high_resolution_clock::now();
        auto durationStandard = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        //Integration with Stratified Sampling
        size_t strataPerDimension = 10;  // Number of layers per dimension
        start = std::chrono::high_resolution_clock::now();
        double resultStratified = integrator.integrateStratified(f, numPoints, threads, strataPerDimension);
        end = std::chrono::high_resolution_clock::now();
        auto durationStratified = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Integration with Hastings
        start = std::chrono::high_resolution_clock::now();
        double resultHastings = integrator.integrateHastings(f, numPoints, threads);
        end = std::chrono::high_resolution_clock::now();
        auto durationHastings = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Results
        std::cout << std::setw(8) << threads
                  << std::setw(15) << durationStandard.count()
                  << std::setw(15) << std::fixed << std::setprecision(6) << resultStandard
                  << std::setw(15) << std::fixed << std::setprecision(6) << resultStratified
                  << std::setw(15) << std::fixed << std::setprecision(6) << resultHastings << "\n";
    }

    return 0;
}
