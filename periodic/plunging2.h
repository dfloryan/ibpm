#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include "ibpm.h"
#include <iterator>

#include "StateVectorNGMRES.h"



#include <cstdlib>
#include <exception>
#include <memory>

#include <Eigen/Core>

#include "Gmres.h"
#include "Newton.h"
#include "NewtonArmijo.h"
#include "exceptions.h"

using namespace ibpm;

enum ModelType { LINEAR, NONLINEAR, ADJOINT, ADJOINT2, LINEARPERIODIC, SFD, INVALID };

// Return the type of model specified in the string modelName
ModelType str2model( string modelName );

// Return the integration scheme specified in the string integratorType
Scheme::SchemeType str2scheme( string integratorType );

/*! \brief Main routine for IBFS code
 *  Set up a timestepper and advance the flow in time.
 */
Eigen::VectorXd plunging(string name, string outdir, int iTecplot, bool TecplotAllGrids, int iRestart, int iForce, int iEnergy, string numDigitInFileName, int nx, int ny, int ngrid, double length, double xOffset, double yOffset, double xShift, double yShift, double alpha, string geomFile, bool ubf, double Reynolds, string modelName, string baseFlow, string icFile, bool resetTime, bool subtractBaseflow, double dt, int numSteps, string integratorType, int period, int periodStart, string periodBaseFlowName, double chi, double Delta) {
    
    cout << "Immersed Boundary Projection Method (IBPM), version "
    << IBPM_VERSION << "\n" << endl;
    
    
    ModelType modelType = str2model( modelName );
    Scheme::SchemeType schemeType = str2scheme( integratorType );
    
    // modify this long if statement?
    if ( ( modelType != NONLINEAR ) && ( modelType != SFD ) ) {
        if (modelType != LINEARPERIODIC && modelType != ADJOINT2 && baseFlow == "" ){
            cout << "ERROR: for linear or adjoint models, "
            "must specify a base flow" << endl;
            exit(1);
        }
        else if (modelType != LINEARPERIODIC && modelType != ADJOINT2 && periodBaseFlowName != ""){
            cout << "WARNING: for linear or adjoint models, "
            "a periodic base flow is not needed" << endl;
            exit(1);
        }
        else if ((modelType == LINEARPERIODIC || modelType == ADJOINT2) && periodBaseFlowName == "" ) {
            cout << "ERROR: for linear periodic model, "
            "must specify a periodic base flow" << endl;
            exit(1);
        }
        else if ((modelType == LINEARPERIODIC || modelType == ADJOINT2) && baseFlow != "" ) {
            cout << "WARNING: for linear periodic model, "
            "a single baseflow is not needed" << endl;
            exit(1);
        }
    }
    
    // create output directory if not already present
    AddSlashToPath( outdir );
    mkdir( outdir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO );
    
    // Name of this run
    cout << "Run name: " << name << "\n" << endl;
    
    // Setup grid
    cout << "Grid parameters:" << endl
    << "    nx      " << nx << endl
    << "    ny      " << ny << endl
    << "    ngrid   " << ngrid << endl
    << "    length  " << length << endl
    << "    xoffset " << xOffset << endl
    << "    yoffset " << yOffset << endl
    << "    xshift  " << xShift << endl
    << "    yshift  " << yShift << endl
    << endl;
    Grid grid( nx, ny, ngrid, length, xOffset, yOffset, xShift, yShift );
    
    // Setup geometry
    Geometry geom;
    cout << "Reading geometry from file " << geomFile << endl;
    if ( geom.load( geomFile ) ) {
        cout << "    " << geom.getNumPoints() << " points on the boundary" << "\n" << endl;
    }
    else {
        exit(-1);
    }
    
    // Setup equations to solve
    cout << "Reynolds number = " << Reynolds << "\n" << endl;
    cout << "Setting up Immersed Boundary Solver..." << flush;
    double magnitude = 1;
    //    double alpha = 0;  // angle of background flow
    double pi = 4. * atan(1.);
    alpha = alpha*pi/180.;
    double xC = 0, yC = 0;
    BaseFlow q_potential( grid, magnitude, alpha );
    // See if unsteady base flow can be used.  Only implemented for a single RigidBody in motion.
    // In the future, have a function geom.ubfEligible() that will make sure that the first RigidBody is moving.
    if( ! geom.isStationary() && (geom.getNumBodies() == 1) && ubf ) {
        Motion* m = geom.transferMotion(); // pull motion from first RigidBody object
        geom.transferCenter(xC,yC);        // pull center of motion from RigidBody object
        q_potential.setMotion( *m );
        q_potential.setCenter(xC,yC);
    }
    if( ubf && (geom.getNumBodies() != 1) ) {
        cout << "Unsteady base flow is only supported for a single moving body.  Exiting program." << endl;
        exit(1);
    }
    
    NavierStokesModel* model = NULL;
    AdjointIBSolver2* adjointsolver = NULL;
    IBSolver* solver = NULL;
    SFDSolver* SFDsolver = NULL;
    State x00( grid, geom.getNumPoints() );
    vector<State> x0(period, x00);
    
    Motion* mot = NULL; // this is for adjoint tests; get rid of later
    switch (modelType){
        case NONLINEAR:
            model =  new NavierStokesModel( grid, geom, Reynolds, q_potential );
            solver = new NonlinearIBSolver( grid, *model, dt, schemeType );
            break;
        case LINEAR:
            if ( ! x00.load( baseFlow ) ) {
                cout << "baseflow failed to load.  Exiting program." << endl;
                exit(1);
            }
            model =  new NavierStokesModel( grid, geom, Reynolds );
            solver = new LinearizedIBSolver( grid, *model, dt, schemeType, x00 );
            break;
        case ADJOINT:
            if ( ! x00.load( baseFlow ) ) {
                cout << "baseflow failed to load.  Exiting program." << endl;
                exit(1);
            }
            model =  new NavierStokesModel( grid, geom, Reynolds );
            solver = new AdjointIBSolver( grid, *model, dt, schemeType, x00 );
            break;
        case ADJOINT2: {
            // load periodic base flow files
            char pbffilename[256];
            string pbf = periodBaseFlowName;
            for (int i = 0; i < period; i++) {
                //cout << "loading phase " << i << " of period-" << period << " base flow:" << endl;
                sprintf(pbffilename, pbf.c_str(), i + periodStart);
                if ( ! x0[i].load(pbffilename) ) {
                    cout << "base flow " << pbffilename << " failed to load. Exiting program." << endl;
                    exit(1);
                }
            }
            x00 = x0[0];
            mot = geom.transferMotion();
            model = new NavierStokesModel( grid, geom, Reynolds );
            adjointsolver = new AdjointIBSolver2( grid, *model, dt, schemeType, x0, period, *mot );
            solver = adjointsolver;
        } break;
        case LINEARPERIODIC:{
            // load periodic baseflow files
            vector<State> x0(period, x00);
            char pbffilename[256];
            string pbf = periodBaseFlowName;
            for (int i=0; i < period; i++) {
                //cout << "loading the " << i << "-th periodic baseflow:" << endl;
                sprintf(pbffilename, pbf.c_str(), i + periodStart);
                x0[i].load(pbffilename);
            }
            x00 = x0[0];
            model =  new NavierStokesModel( grid, geom, Reynolds );
            solver = new LinearizedPeriodicIBSolver( grid, *model, dt, schemeType, x0, period ) ;
            break;
        }
        case SFD:{
            cout << "SFD parameters:" << endl;
            cout << "    chi =   " << chi << endl;
            cout << "    Delta = " << Delta << endl << endl;
            model =  new NavierStokesModel( grid, geom, Reynolds, q_potential );
            SFDsolver = new SFDSolver( grid, *model, dt, schemeType, Delta, chi ) ;
            solver = SFDsolver;
            break;
        }
        case INVALID:
            cout << "ERROR: must specify a valid modelType" << endl;
            exit(1);
            break;
    }
    
    assert( model != NULL );
    assert( solver != NULL );
    if( modelType == SFD ) {
        assert( chi != 0 );
        assert( SFDsolver != NULL );
    }
    // NOTE: still need to initialize model, but wait until after loading the initial
    //       condition, so we know what the initial time is, for moving the bodies
    
    // Load initial condition
    State x( grid, geom.getNumPoints() );
    x.omega = 0.;
    x.f = 0.;
    x.q = 0.;
    if (icFile != "") {
        cout << "Loading initial condition from file: " << icFile << endl;
        if ( ! x.load(icFile) ) {
            cout << "    (failed: using zero initial condition)" << endl;
        }
        if ( subtractBaseflow == true ) {
            cout << "    Subtracting initial condition by baseflow to form a linear initial perturbation" << endl;
            if (modelType != NONLINEAR) {
                assert((x.q).Ngrid() == (x00.q).Ngrid());
                assert((x.omega).Ngrid() == (x00.omega).Ngrid());
                x.q -= x00.q;
                x.omega -= x00.omega;
                x.f = 0;
            }
            else {
                cout << "Flag subbaseflow should be true only for linear cases"<< endl;
                exit(1);
            }
        }
        
        if ( modelType == SFD ) {
            SFDsolver->loadFilteredState( icFile );
        }
        
    }
    else {
        cout << "Using zero initial condition" << endl;
    }
    
    if (resetTime == true) {
        x.timestep = 0;
        x.time = 0.;
    }
    
    // update the geometry to the current time
    geom.moveBodies( x.time );
    
    // Initialize model and timestepper
    model->init();
    cout << "using " << solver->getName() << " timestepper" << endl;
    cout << "    dt = " << dt << "\n" << endl;
    if ( ! solver->load( outdir + name ) ) {
        // Set the tolerance for a ConjugateGradient solver below
        // Otherwise default is tol = 1e-7
        // solver->setTol( 1e-8 )
        solver->init();
        solver->save( outdir + name );
    }
    
    // Calculate flux for state, in case only vorticity was saved
    if( ! q_potential.isStationary() ) {
        q_potential.setAlphaMag( x.time );
        alpha = q_potential.getAlpha();
    }
    model->updateOperators( x.time );
    model->refreshState( x );
    
    cout << endl << "Initial timestep = " << x.timestep << "\n" << endl;
    
    // Setup output routines
    OutputTecplot tecplot( outdir + name + numDigitInFileName + ".plt", "Test run, step" +  numDigitInFileName, TecplotAllGrids);
    if(TecplotAllGrids) tecplot.setFilename( outdir + name + numDigitInFileName + "_g%01d.plt" );
    OutputRestart restart( outdir + name + numDigitInFileName + ".bin" );
    OutputForce force( outdir + name + ".force" );
    OutputEnergy energy( outdir + name + ".energy" );
    
    Logger logger;
    // Output Tecplot file every timestep
    if ( iTecplot > 0 ) {
        cout << "Writing Tecplot file every " << iTecplot << " step(s)" << endl;
        logger.addOutput( &tecplot, iTecplot );
    }
    if ( iRestart > 0 ) {
        cout << "Writing restart file every " << iRestart << " step(s)" << endl;
        logger.addOutput( &restart, iRestart );
    }
    if ( iForce > 0 ) {
        cout << "Writing forces every " << iForce << " step(s)" << endl;
        logger.addOutput( &force, iForce );
    }
    if ( iEnergy > 0 ) {
        cout << "Writing energy every " << iForce << " step(s)" << endl;
        logger.addOutput( &energy, iEnergy );
    }
    cout << endl;
    logger.init();
    
    // Create StateVector object equal to State x
    StateVector xv(x);
    
    // Create a map to evolve state forward in time by numSteps steps
    StateVectorNGMRES mapper( solver, numSteps );
    
    // Map state forward a few times to create better initial guess for Newton solver
    for(int i=0; i<3; i++) {
        mapper.mapForward(xv);
    }
    
    linear_solver_ns::Gmres<StateVector> gmres(StateVectorInnerProduct, 5, 1.e-6, true);
    double tol = 1.e-6;
    int max_iter = 10;
    double jacobian_dx = 1.e-3;
    std::unique_ptr<newton_ns::Newton<StateVector>> newton;
    newton.reset(new newton_ns::Newton<StateVector>{mapper, gmres, StateVectorNorm, tol, max_iter, jacobian_dx, true});
    //newton.reset(new newton_ns::NewtonArmijo<StateVector>{mapper, gmres, StateVectorNorm, tol, max_iter, jacobian_dx, 1, 10, 0.1, 0.5, 1.e-4, true});
    try {
        newton->solve(xv);
    }
    catch (newton_ns::NewtonError& ne) {
        cerr << "Newton error: " << ne.what() << endl;
        exit(2);
    }
    catch (std::exception& e) {
        cerr << "Other exception: " << e.what() << endl;
        exit(3);
    }
    
    // Save a period of restart files and forces if nonlinear, or gradient if adjoint
    // Return mean drag if nonlinear, gradient if linear
    Eigen::VectorXd objgrad;
    if(modelType == NONLINEAR) {
        xv.x.timestep = 0; // reset time
        xv.x.time = 0.;
        logger.doOutput( q_potential, xv.x );
        double xF, yF; // forces in x and y direction (same as drag,lift if alpha=0)
        xv.x.computeNetForce( xF, yF );
        objgrad.resize(1);
        objgrad[0] = xF;
        for(int i = 0; i < numSteps - 1; i++) {
            solver->advance(xv.x);
            xv.x.computeNetForce( xF, yF );
            objgrad[0] += xF;
            logger.doOutput( q_potential, xv.x );
        }
        objgrad *= dt;
    } else if( modelType == ADJOINT2 ) {
        int k;
        objgrad.resize(numSteps); // vector to store gradient
        double dx2 = grid.Dx() * grid.Dx();
        
        for(int i = 0; i < numSteps; i++) {
            solver->advance(xv.x);
            
            //k = period - xv.x.timestep - 1; // if not periodic
            k = (period - (xv.x.timestep % period)) % period; // if periodic
            double partial = 0.;
            Flux dqdp(x0[k].q);
            TangentSE2 gg(0, 0, 0, 0, -1., 0);
            dqdp.setFlow(gg, 0, 0);
            double deriv1 = (-InnerProduct(dqdp, CrossProduct(xv.x.q, x0[k].omega))) / dx2;
            BoundaryVector dudp(xv.x.f.getNumPoints());
            dudp = - model->flux2boundary(dqdp);
            double deriv2 = InnerProduct(dudp, xv.x.f);
            
            objgrad[i] = partial + deriv1 + deriv2;
        }
        
        // Flip gradient
        objgrad.reverseInPlace();
    }
    
    logger.cleanup();
    
    delete solver;
    return objgrad;
}

ModelType str2model( string modelName ) {
    ModelType type;
    MakeLowercase( modelName );
    
    if ( modelName == "nonlinear" ) {
        type = NONLINEAR;
    }
    else if ( modelName == "linear" ) {
        type = LINEAR;
    }
    else if ( modelName == "adjoint" ) {
        type = ADJOINT;
    }
    else if ( modelName == "adjoint2" ) {
        type = ADJOINT2;
    }
    else if ( modelName == "linearperiodic" ) {
        type = LINEARPERIODIC;
    }
    else if ( modelName == "sfd" ) {
        type = SFD;
    }
    else {
        cerr << "Unrecognized model: " << modelName << endl;
        type = INVALID;
    }
    return type;
}

Scheme::SchemeType str2scheme( string schemeName ) {
    Scheme::SchemeType type;
    MakeLowercase( schemeName );
    if ( schemeName == "euler" ) {
        type = Scheme::EULER;
    }
    else if ( schemeName == "ab2" ) {
        type = Scheme::AB2;
    }
    else if ( schemeName == "rk3" ) {
        type = Scheme::RK3;
    }
    else if ( schemeName == "rk3b" ) {
        type = Scheme::RK3b;
    }
    else {
        cerr << "Unrecognized integration scheme: " << schemeName;
        cerr << "    Exiting program." << endl;
        exit(1);
    }
    return type;
}


