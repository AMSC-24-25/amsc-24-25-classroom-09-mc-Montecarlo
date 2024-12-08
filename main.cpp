#include <random>
#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <thread>
#include <future>
#include <chrono>
#include <iomanip>
#include <Eigen/Dense>

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

/* H form -> P = {x ∈ R^n ∣ A⋅x≤b }
 where every row of Ax≤b represents an half-space*/
class Polytope : public IntegrationDomain {

    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::pair<Eigen::VectorXd, Eigen::VectorXd> bounds;

    public Polytope(const Eigen::MatrixXd &A_ , const Eigen::VectorXd b_, const std::pair<Eigen::VectorXd, Eigen::VectorXd> &explicitBounds)
    : A(A_), b(b_), bounds(explicitBounds) {
        if(A.rows()! = b.size()){ // A's rows must be equal to b's rows
            throw std::invalid_argument("A and b's dimensions aren't compatible")
        }
         if (explicitBounds.first.size() != A.cols() || explicitBounds.second.size() != A.cols()) {
            throw std::invalid_argument("Bounds must match the dimensionality of the polytope.");
        } // ensures that the dimensionality of the bounds matches the columns of A
    }

   // verifies if a point belongs to P and returns true if it does, false otherwise
     bool contains(const Eigen::VectorXd &point) const override {
      Eigen::VectorXd pointVec = Eigen::Map<const Eigen::VectorXd>(point.data(), point.size());
        /* A*point is a mvm whose result is confronted with the
        corresponding element of b*/
        return ((A*pointVec).array() <= b.array()).all(); // this is done for all the points
        
     }

   // takes the bound as a parameter 
    std::vector<std::pair<double, double>> getBounds() const override {
        std::vector<std::pair<double, double>> box(bounds.first.size());
        for (size_t i = 0; i < bounds.first.size(); ++i) {
            box[i] = {bounds.first[i], bounds.second[i]};
        }
        return box;
    }


    double getVolume(size_t numSamples, size_t strataPerDimension) const {
    auto [lower, upper] = bounds; // Bounding box
    size_t dimensions = lower.size();

    // Calculate bounding box volume
    double boundingBoxVolume = 1.0;
    for (size_t i = 0; i < dimensions; ++i) {
        boundingBoxVolume *= (upper[i] - lower[i]);
    }

    // Monte Carlo sampling with stratification
    size_t numThreads = std::thread::hardware_concurrency();
    size_t pointsPerThread = numSamples / numThreads;

    // Lambda for thread workers
    auto worker = [&](size_t threadIndex) {
        std::mt19937 rng(std::random_device{}());
        double localCount = 0;

        for (size_t i = 0; i < pointsPerThread; ++i) {
            auto point = generateStratifiedPoint(rng, strataPerDimension); // Use existing function
            if (contains(point)) {
                localCount++;
            }
        }

        return localCount;
    };

    // Parallel execution
    std::vector<std::future<double>> futures;
    for (size_t i = 0; i < numThreads; ++i) {
        futures.emplace_back(std::async(std::launch::async, worker, i));
    }

    // Aggregate results
    double pointsInside = 0.0;
    for (auto &future : futures) {
        pointsInside += future.get();
    }

    // Estimate volume
    return boundingBoxVolume * (pointsInside / numSamples);
}

}


class MonteCarloIntegrator {
    std::vector<std::uniform_real_distribution<double>> distributions;                  //distributions: a vector of uniform distributions. Each distribution generates random numbers in a specific range, one for each domain size
    std::vector<std::mt19937> engines; // TODO: use different types of engines?         //vector of random number generators
    const IntegrationDomain &domain;                                                    //integration domain

    // Initializes pseudo-random engines with seeds from std::seed_seq
    // std::seed_seq is initialized with non-deterministic random values from std::random_device

    //--> Method to initialize engines with different types and seeds
    void initializeEngines(size_t numThreads, std::string engineType) {
        if (engineType == "mt19937") {
            // Use Mersenne Twister
            std::random_device rd;
            std::vector<std::uint32_t> entropy(numThreads);
            for (size_t i = 0; i < numThreads; ++i) {
                entropy[i] = rd();
            }
            std::seed_seq seq(entropy.begin(), entropy.end());
            std::vector<std::uint32_t> seeds(numThreads);
            seq.generate(seeds.begin(), seeds.end());

            engines.clear();
            for (auto seed : seeds) {
                engines.emplace_back(std::mt19937(seed));
            }
        } else if (engineType == "xorshift") {
            // Use Xorshift128+
            std::random_device rd;
            std::array<std::uint32_t, 4> seedData = {rd(), rd(), rd(), rd()};
            std::seed_seq seedSeq(seedData.begin(), seedData.end());

            engines.clear();
            for (size_t i = 0; i < numThreads; ++i) {
                engines.emplace_back(std::mt19937_64(seedSeq));
            }
        } else {
            throw std::invalid_argument("Invalid engine type");
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


};

// TODO: `Use different type of sampling: stratified sampling, importance sampling (Metropolis-Hastings).`
//-> stratified sampling


//-> Metropolis-Hastings

/*
Metropolis-Hastings is useful to get a sample generated from a "complex" probability distribution.
How does it work: 
-choose a starting point x0 (we can choose whatever we want, like the origin)
-define a proposal distribution Q(x'|x), it will generates x' starting from x
-x' is the new candidate point
-compute r (acceptance rateo)
-decide to accept the point or not
-repeat

//To parallelize it: divide the sample in multiple chains and compute the avg

*/


// Metropolis-Hastings Integrator class with parallel chains
class MetropolisHastingsIntegrator {
    const IntegrationDomain &domain; 
    std::normal_distribution<double> proposalDist; // Gaussian proposal distribution

    // Initializes random engines with unique seeds for each chain
    std::vector<std::mt19937> initializeEngines(size_t numChains) {
        std::random_device rd; 
        std::vector<std::mt19937> engines(numChains); 
        for (size_t i = 0; i < numChains; ++i) {
            engines[i] = std::mt19937(rd()); // Initialize each engine with a unique seed.
        }
        return engines;
    }

public:
    // Constructor to initialize the integrator with a domain
    explicit MetropolisHastingsIntegrator(const IntegrationDomain &d)
        : domain(d), proposalDist(0.0, 1.0) {} // Gaussian proposal distribution (mean = 0, stddev = 1).

    // Perform a single Metropolis-Hastings chain
    double integrateSingleChain(
        const std::function<double(const std::vector<double> &)> &f, 
        size_t numPoints,                                            
        const std::vector<double> &initialPoint,                     
        std::mt19937 &engine,                                        
        size_t &accepted                                             // Counter for accepted points.
    ) {
        size_t dimensions = initialPoint.size(); // Number of dimensions of the domain.
        std::vector<double> currentPoint = initialPoint; // Current point (initially the starting point).
        double currentFValue = f(currentPoint); // Value of the function at the current point.

        double sumF = 0.0; .
        accepted = 0;

        for (size_t i = 0; i < numPoints; ++i) {
            // Generate a candidate point by adding Gaussian noise to the current point.
            std::vector<double> candidatePoint(dimensions);
            for (size_t d = 0; d < dimensions; ++d) {
                candidatePoint[d] = currentPoint[d] + proposalDist(engine);
            }

            // Check if the candidate point is within the domain.
            if (domain.contains(candidatePoint)) {
                double candidateFValue = f(candidatePoint);

                // Compute the Metropolis-Hastings acceptance ratio.
                double acceptanceRatio = candidateFValue / currentFValue;

                // Accept or reject the candidate point based on a uniform random threshold.
                if (std::uniform_real_distribution<>(0.0, 1.0)(engine) <= acceptanceRatio) {
                    currentPoint = candidatePoint; 
                    currentFValue = candidateFValue; 
                    ++accepted;
                }
            }

            // Add the current function value to the cumulative sum.
            sumF += currentFValue;
        }

        // Return the partial integral result scaled by the domain's volume.
        return domain.getVolume() * sumF / static_cast<double>(numPoints);
    }

    // Perform parallel integration with multiple chains
    double integrateParallel(
        const std::function<double(const std::vector<double> &)> &f, 
        size_t numPoints,                                           
        const std::vector<double> &initialPoint,                     
        size_t numChains                                            
    ) {
        // Initialize random engines for each chain.
        auto engines = initializeEngines(numChains);

        // Calculate the number of points to be sampled per chain.
        size_t pointsPerChain = numPoints / numChains;

        // Launch parallel chains using std::async.
        std::vector<std::future<std::pair<double, size_t>>> futures;
        for (size_t i = 0; i < numChains; ++i) {
            futures.push_back(std::async(
                std::launch::async, // Execute asynchronously.
                [this, &f, pointsPerChain, &initialPoint, &engine = engines[i]]() {
                    size_t accepted = 0; // Counter for accepted points in this chain.
                    double result = this->integrateSingleChain(f, pointsPerChain, initialPoint, engine, accepted);
                    return std::make_pair(result, accepted); // Return the partial result and accepted points.
                }
            ));
        }

        // Collect results from all chains.
        double totalResult = 0.0; // Sum of partial results from all chains.
        size_t totalAccepted = 0; // Sum of accepted points from all chains.
        for (auto &future : futures) {
            auto [result, accepted] = future.get(); // Retrieve the result from each chain.
            totalResult += result; 
            totalAccepted += accepted;
        }

        // Print the overall acceptance rate.
        std::cout << "Overall acceptance rate: "
                  << static_cast<double>(totalAccepted) / numPoints * 100.0 << "%\n";

        // Return the combined result (average of chain results).
        return totalResult / numChains;
    }
};



//TODO: update the main with the stratified and MH samplings

int main() {
    // Function to integrate: x^2 + y^2
    auto f = [](const std::vector<double> &x) {
        return x[0] * x[0] + x[1] * x[1];                                       //function
    };

    // A circle with radius 1 with center in (0, 0)
    Hypersphere sphere(2, 1.0);                                                //Integration domain sphere(dimension, radius)
    MonteCarloIntegrator integrator(sphere);

    const std::vector<size_t> threadCounts = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    constexpr size_t numPoints = 10000000; // 10^7                            //total number of random points generated for Monte Carlo integration

    std::cout << "Integration with " << numPoints << " points\n\n";
    std::cout << std::setw(8) << "Threads"
            << std::setw(15) << "Time (µs)"
            << std::setw(15) << "Result" << "\n";
    std::cout << std::string(38, '-') << "\n";

    // Must show an increase of speed until a certain point (which is optimal #threads on your machine)
    for (size_t threads: threadCounts) {
        auto start = std::chrono::high_resolution_clock::now();                 //to record the time before and after the calculation
        double result = integrator.integrate(f, numPoints, threads);
        auto end = std::chrono::high_resolution_clock::now();                   //Calculate the total time spent integrating in microseconds
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Must be close to pi/2
        std::cout << std::setw(8) << threads
                << std::setw(15) << duration.count()
                << std::setw(15) << std::fixed << std::setprecision(6) << result << "\n";
    }

    // TODO: find best #threads and iterate over increasing #points to show improvement in precision?

    return 0;
}