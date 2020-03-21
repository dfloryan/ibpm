#include <iostream>
#include <iomanip>
#include <fstream>
#include "meta.h"
#include "boundedproblem.h"
#include "problem.h"
#include "lbfgsbsolver.h"
#include "plunging2.h"

#define MAXBUFSIZE  ((int) 1e6)

using namespace ibpm;

Eigen::VectorXd cumtrapz(const Eigen::VectorXd & t, const Eigen::VectorXd & ydot)
{
    int n = t.rows();
    double dt = t[1] - t[0];
    Eigen::VectorXd y = Eigen::VectorXd::Zero(n);
    for(int i = 1; i < n; i++) {
        y[i] = y[i - 1] + (ydot[i] + ydot[i - 1])*dt/2.;
    }
    return y;
};

Eigen::MatrixXd readMatrix(const char *filename)
{
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];
    
    // Read numbers from file into buffer.
    std::ifstream infile;
    infile.open(filename);
    
    // Skip first row of file
    std::string line;
    getline(infile, line);
    
    // Read the rest of the file
    while (! infile.eof())
    {
        getline(infile, line);
        int temp_cols = 0;
        std::stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];
        
        if (temp_cols == 0)
            continue;
        
        if (cols == 0)
            cols = temp_cols;
        rows++;
    }
    
    infile.close();
    
    rows--;
    
    // Populate matrix with numbers.
    Eigen::MatrixXd result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];
    
    return result;
};

void writeMatrix(const Eigen::VectorXd & t, const Eigen::VectorXd & ydot, const char *filename)
{
    // Assuming purely heaving motion here
    double dt = t[1] - t[0];
    int n = t.rows();
    
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(n + 1, 7);
    
    mat.block(0, 0, n, 1) = t;
    mat(n, 0) = dt + t[n - 1];
    
    mat.block(0, 5, n, 1) = ydot;
    mat(n, 5) = ydot[0];
    
    mat.block(0, 2, n + 1, 1) = cumtrapz(mat.block(0, 0, n + 1, 1), mat.block(0, 5, n + 1, 1));
    
    std::ofstream filesaver(filename);
    if (filesaver.is_open())
    {
        filesaver << n + 1 << std::endl;
        filesaver << std::fixed << mat << std::endl;
    }
};

// we define a new problem for optimizing the Simple function
// we use a templated-class rather than "auto"-lambda function for a clean architecture
template<typename T>
class Simple : public cppoptlib::BoundedProblem<T> {
  public:
    //using typename cppoptlib::BoundedProblem<T>::TVector; // Inherit the Vector typedef
    using Superclass = cppoptlib::BoundedProblem<T>;
    using typename Superclass::TVector;
    using TMatrix = typename Superclass::THessian;
    
    string motionfile; // file containing motion
    Eigen::VectorXd t; // time vector
    
    // Output parameters
    string nameNonlinear;
    string nameAdjoint;
    string outdir;
    int iTecplot;
    bool TecplotAllGrids;
    int iRestartNonlinear;
    int iRestartAdjoint;
    int iForceNonlinear;
    int iForceAdjoint;
    int iEnergy;
    string numDigitInFileName;
    
    // Grid parameters
    int nx;
    int ny;
    int ngrid;
    double length;
    double xOffset;
    double yOffset;
    double xShift;
    double yShift;
    double alpha;
    
    // Simulation parameters
    string geomFile;
    bool ubfNonlinear;
    bool ubfAdjoint;
    double Reynolds;
    string modelNameNonlinear;
    string modelNameAdjoint;
    string baseFlow;
    
    // Initial condition
    string icFileNonlinear;
    string icFileAdjoint;
    bool resetTime;
    bool subtractBaseflow;
    
    // Integration parameters
    double dt;
    int numSteps;
    string integratorType;
    
    // Linear-periodic model
    int period;
    int periodStart;
    string periodBaseFlowName;
    
    // SFD
    double chi;
    double Delta;
    
    // Constructor
    Simple(int dim, int argc, char* argv[]) : Superclass(dim) {
        // Get parameters
        ParmParser parser( argc, argv );
        bool helpFlag = parser.getFlag( "h", "print this help message and exit" );
        
        motionfile = parser.getString( "motionfile", "file containing motion", "" );
        Eigen::MatrixXd testmotion = readMatrix(motionfile.c_str());
        t = testmotion.block(0, 0, testmotion.rows() - 1, 1);
        
        // Output parameters
        nameNonlinear = parser.getString( "nameNonlinear", "nonlinear run name", "ibpm" );
        nameAdjoint = parser.getString( "nameAdjoint", "adjoint run name", "ibpm" );
        outdir = parser.getString( "outdir", "directory for saving output", "." );
        iTecplot = parser.getInt( "tecplot", "if >0, write a Tecplot file every n timesteps", 100);
        TecplotAllGrids = parser.getBool( "tecplotallgrids", "Tecplot output for all grids, or not", false );
        iRestartNonlinear = parser.getInt( "restartNonlinear", "if >0, write a nonlinear restart file every n timesteps", 100);
        iRestartAdjoint = parser.getInt( "restartAdjoint", "if >0, write an adjoint restart file every n timesteps", 100);
        iForceNonlinear = parser.getInt( "forceNonlinear", "if >0, write nonlinear forces every n timesteps", 1);
        iForceAdjoint = parser.getInt( "forceAdjoint", "if >0, write adjoint forces every n timesteps", 1);
        iEnergy = parser.getInt( "energy", "if >0, write energy every n timesteps", 0);
        numDigitInFileName = parser.getString( "numdigfilename", "number of digits for time representation in filename", "%05d");
        
        // Grid parameters
        nx = parser.getInt( "nx", "number of gridpoints in x-direction", 200 );
        ny = parser.getInt( "ny", "number of gridpoints in y-direction", 200 );
        ngrid = parser.getInt( "ngrid", "number of grid levels for multi-domain scheme", 1 );
        length = parser.getDouble( "length", "length of finest domain in x-dir", 4.0 );
        xOffset = parser.getDouble( "xoffset", "x-coordinate of left edge of finest domain", -2. );
        yOffset = parser.getDouble( "yoffset", "y-coordinate of bottom edge of finest domain", -2. );
        xShift = parser.getDouble( "xshift", "percentage offset between grid levels in x-direction", 0. );
        yShift = parser.getDouble( "yshift", "percentage offset between grid levels in y-direction", 0. );
        alpha = parser.getDouble( "alpha", "angle of attack of base flow", 0.);
        
        // Simulation parameters
        geomFile = parser.getString( "geom", "filename for reading geometry", nameNonlinear + ".geom" );
        ubfNonlinear = parser.getBool( "ubfNonlinear", "Use unsteady base flow, or not", false );
        ubfAdjoint = parser.getBool( "ubfAdjoint", "Use unsteady base flow, or not", false );
        Reynolds = parser.getDouble("Re", "Reynolds number", 100.);
        modelNameNonlinear = parser.getString( "modelNonlinear", "type of model (linear, nonlinear, adjoint, adjoint2, linearperiodic, sfd)", "nonlinear" );
        modelNameAdjoint = parser.getString( "modelAdjoint", "type of model (linear, nonlinear, adjoint, adjoint2, linearperiodic, sfd)", "nonlinear" );
        baseFlow = parser.getString( "baseflow", "base flow for linear/adjoint/adjoint2 model", "" );
        
        // Initial condition
        icFileNonlinear = parser.getString( "icNonlinear", "nonlinear initial condition filename", "");
        icFileAdjoint = parser.getString( "icAdjoint", "adjoint initial condition filename", "");
        resetTime = parser.getBool( "resettime", "Reset time when subtracting ic by baseflow (1/0(true/false))", false);
        subtractBaseflow = parser.getBool( "subbaseflow", "Subtract ic by baseflow (1/0(true/false))", false);
        
        // Integration parameters
        dt = parser.getDouble( "dt", "timestep", 0.02 );
        numSteps = parser.getInt( "nsteps", "number of timesteps to compute", 250 );
        integratorType = parser.getString( "scheme", "timestepping scheme (euler,ab2,rk3,rk3b)", "rk3" );
        
        // Linear-periodic model
        period = parser.getInt( "period", "period of periodic baseflow", 1);
        periodStart = parser.getInt( "periodstart", "start time of periodic baseflow", 0);
        periodBaseFlowName = parser.getString( "pbaseflowname", "name of periodic baseflow, e.g. 'flow/ibpmperiodic%05d.bin', with '%05d' as time, decided by periodstart/period", "" );
        
        // SFD
        chi = parser.getDouble( "chi", "sfd gain", 0.02 );
        Delta = parser.getDouble( "Delta", "sfd cutoff frequency", 15. );
        
        if ( ! parser.inputIsValid() || helpFlag ) {
            parser.printUsage( cerr );
            exit(1);
        }
        
        // output command line arguments
        string cmd = parser.getParameters();
        cout << "Command:" << endl << cmd << "\n" << endl;
    }

    // this is just the objective (NOT optional)
    T value(const TVector &x) {
        // Write motion to file
        writeMatrix(t, x, motionfile.c_str());
        std::cout << "\n\nydot is\n\n" << std::endl;
        std::cout << x << std::endl;
        
        // Run forward simulation
        Eigen::VectorXd a = plunging(nameNonlinear, outdir, iTecplot, TecplotAllGrids, iRestartNonlinear, iForceNonlinear, iEnergy, numDigitInFileName, nx, ny, ngrid, length, xOffset, yOffset, xShift, yShift, alpha, geomFile, ubfNonlinear, Reynolds, modelNameNonlinear, baseFlow, icFileNonlinear, resetTime, subtractBaseflow, dt, numSteps, integratorType, period, periodStart, periodBaseFlowName, chi, Delta);
        std::cout << "\n with drag = " << a[0] << std::endl;
        return a[0];
    }

    // if you calculated the derivative by hand
    // you can implement it here (OPTIONAL)
    // otherwise it will fall back to (bad) numerical finite differences
    void gradient(const TVector &x, TVector &grad) {
        grad = plunging(nameAdjoint, outdir, iTecplot, TecplotAllGrids, iRestartAdjoint, iForceAdjoint, iEnergy, numDigitInFileName, nx, ny, ngrid, length, xOffset, yOffset, xShift, yShift, alpha, geomFile, ubfAdjoint, Reynolds, modelNameAdjoint, baseFlow, icFileAdjoint, resetTime, subtractBaseflow, dt, numSteps, integratorType, period, periodStart, periodBaseFlowName, chi, Delta);
        
        // Zero the mean of the gradient for now
        grad -= Eigen::VectorXd::Ones(grad.size())*grad.mean();
        
        // Print to screen
        std::cout << "\n\ngradient is\n\n" << std::endl;
        std::cout << grad << std::endl;
    }

    bool callback(const cppoptlib::Criteria<T> &state, const TVector &x) {
        std::cout << "(" << std::setw(2) << state.iterations << ")"
                  << " ||dx|| = " << std::fixed << std::setw(8) << std::setprecision(4) << state.gradNorm
                  << " ||x|| = "  << std::setw(6) << x.norm()
                  << " f(x) = "   << std::setw(8) << value(x)
                  << " x = [" << std::setprecision(8) << x.transpose() << "]" << std::endl;
        
        // Write motion from current iteration to file
        std::string filename = "iteration";
        filename += std::to_string(state.iterations);
        filename += ".txt";
        writeMatrix(t, x, filename.c_str());
        
        return true;
    }
};

int main(int argc, char *argv[]) {
    typedef Simple<double> TSimple;
    int dim = 200; // hard-coded dimension. Will have to change this in the future
    TSimple f(dim, argc, argv);
    
    // Load initial motion
    Eigen::MatrixXd dataMat = readMatrix(f.motionfile.c_str());
    Eigen::VectorXd ydot = dataMat.block(0, 5, dataMat.rows() - 1, 1);
    
    // Create bounds
    f.setLowerBound(Eigen::VectorXd::Ones(dim) * ydot.minCoeff());
    f.setUpperBound(Eigen::VectorXd::Ones(dim) * ydot.maxCoeff());

    // Set solver criteria
    cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
    crit.iterations = 10;
    
    // Create solver
    cppoptlib::LbfgsbSolver<TSimple> solver;
    solver.setStopCriteria(crit);
    
    // Perform optimization
    solver.minimize(f, ydot);
    std::cout << "f in argmin " << f(ydot) << std::endl;
    std::cout << "Solver status: " << solver.status() << std::endl;
    std::cout << "Final criteria values: " << std::endl << solver.criteria() << std::endl;
    return 0;
}
