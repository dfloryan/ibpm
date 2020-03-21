#include <iostream>
#include <iomanip>
#include <Eigen/Core>

#include "plunging2.h"

int main(int argc, char* argv[]) {
    Eigen::VectorXd a;
    
    // Get parameters
    ParmParser parser( argc, argv );
    bool helpFlag = parser.getFlag( "h", "print this help message and exit" );
    
    // Output parameters
    string name = parser.getString( "name", "run name", "ibpm" );
    string outdir = parser.getString( "outdir", "directory for saving output", "." );
    int iTecplot = parser.getInt( "tecplot", "if >0, write a Tecplot file every n timesteps", 100);
    bool TecplotAllGrids = parser.getBool( "tecplotallgrids", "Tecplot output for all grids, or not", false );
    int iRestart = parser.getInt( "restart", "if >0, write a restart file every n timesteps", 100);
    int iForce = parser.getInt( "force", "if >0, write forces every n timesteps", 1);
    int iEnergy = parser.getInt( "energy", "if >0, write energy every n timesteps", 0);
    string numDigitInFileName = parser.getString( "numdigfilename", "number of digits for time representation in filename", "%05d");
    
    // Grid parameters
    int nx = parser.getInt( "nx", "number of gridpoints in x-direction", 200 );
    int ny = parser.getInt( "ny", "number of gridpoints in y-direction", 200 );
    int ngrid = parser.getInt( "ngrid", "number of grid levels for multi-domain scheme", 1 );
    double length = parser.getDouble( "length", "length of finest domain in x-dir", 4.0 );
    double xOffset = parser.getDouble( "xoffset", "x-coordinate of left edge of finest domain", -2. );
    double yOffset = parser.getDouble( "yoffset", "y-coordinate of bottom edge of finest domain", -2. );
    double xShift = parser.getDouble( "xshift", "percentage offset between grid levels in x-direction", 0. );
    double yShift = parser.getDouble( "yshift", "percentage offset between grid levels in y-direction", 0. );
    double alpha = parser.getDouble( "alpha", "angle of attack of base flow", 0.);
    
    // Simulation parameters
    string geomFile = parser.getString( "geom", "filename for reading geometry", name + ".geom" );
    bool ubf = parser.getBool( "ubf", "Use unsteady base flow, or not", false );
    double Reynolds = parser.getDouble("Re", "Reynolds number", 100.);
    string modelName = parser.getString( "model", "type of model (linear, nonlinear, adjoint, adjoint2, linearperiodic, sfd)", "nonlinear" );
    string baseFlow = parser.getString( "baseflow", "base flow for linear/adjoint/adjoint2 model", "" );
    
    // Initial condition
    string icFile = parser.getString( "ic", "initial condition filename", "");
    bool resetTime = parser.getBool( "resettime", "Reset time when subtracting ic by baseflow (1/0(true/false))", false);
    bool subtractBaseflow = parser.getBool( "subbaseflow", "Subtract ic by baseflow (1/0(true/false))", false);
    
    // Integration parameters
    double dt = parser.getDouble( "dt", "timestep", 0.02 );
    int numSteps = parser.getInt( "nsteps", "number of timesteps to compute", 250 );
    string integratorType = parser.getString( "scheme", "timestepping scheme (euler,ab2,rk3,rk3b)", "rk3" );
    
    // Linear-periodic model
    int period = parser.getInt( "period", "period of periodic baseflow", 1);
    int periodStart = parser.getInt( "periodstart", "start time of periodic baseflow", 0);
    string periodBaseFlowName = parser.getString( "pbaseflowname", "name of periodic baseflow, e.g. 'flow/ibpmperiodic%05d.bin', with '%05d' as time, decided by periodstart/period", "" );
    
    // SFD
    double chi = parser.getDouble( "chi", "sfd gain", 0.02 );
    double Delta = parser.getDouble( "Delta", "sfd cutoff frequency", 15. );
    
    
    ModelType modelType = str2model( modelName );
    
    if ( ! parser.inputIsValid() || modelType == INVALID || helpFlag ) {
        parser.printUsage( cerr );
        exit(1);
    }
    
    // output command line arguments
    string cmd = parser.getParameters();
    cout << "Command:" << endl << cmd << "\n" << endl;
    
    a = plunging(name, outdir, iTecplot, TecplotAllGrids, iRestart, iForce, iEnergy, numDigitInFileName, nx, ny, ngrid, length, xOffset, yOffset, xShift, yShift, alpha, geomFile, ubf, Reynolds, modelName, baseFlow, icFile, resetTime, subtractBaseflow, dt, numSteps, integratorType, period, periodStart, periodBaseFlowName, chi, Delta);
    cout << "output is " << a << endl;
    return 0;
}
