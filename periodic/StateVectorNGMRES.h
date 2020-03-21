#ifndef STATEVECTORNGMRES_H
#define STATEVECTORNGMRES_H

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "IBSolver.h"
#include "BaseFlow.h"
#include "Logger.h"
#include "StateVector.h"
#include "VectorOperations.h"

namespace ibpm{
    
// Inner product between StateVectors (over vorticity)
inline double StateVectorInnerProduct(const StateVector& u, const StateVector& v) {
    return InnerProduct(u.x.omega, v.x.omega );
}
    
// Norm of a StateVector
inline double StateVectorNorm(const StateVector& u) {
    return sqrt(StateVectorInnerProduct(u, u));
}
    
class StateVectorNGMRES{
public:
    // Constructor
    StateVectorNGMRES(
        IBSolver* solver,
        int nsteps
        ) :
        _solver( solver ),
        _nsteps( nsteps )
        {}
    
    // Function to map StateVector nsteps forward
    void mapForward( StateVector& u) {
        std::cout << "Integrating for " << _nsteps << " steps" << std::endl;
        for(int i=1; i <= _nsteps; ++i) {
            _solver->advance( u.x );
        }
    }
    
    // Overload () operator: map state forward and return difference
    StateVector operator() ( const StateVector& u ) {
        StateVector v(u);
        mapForward(v);
        v -= u;
        return v;
    }
    
private:
    IBSolver* _solver;
    int _nsteps;
};
    
} // ibpm

#endif // STATEVETORNGMRES_H
