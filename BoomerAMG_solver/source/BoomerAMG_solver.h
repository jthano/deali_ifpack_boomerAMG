#include<Ifpack_Hypre.h>
#include<Epetra_MultiVector.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/base/config.h>

DEAL_II_NAMESPACE_OPEN

namespace TrilinosWrappers
{

class SolverBoomerAMG{
public:
    struct AdditionalData
    {
    	//
    	// TODO: Some of this belongs to solver control
    	//
    	explicit AdditionalData(
    			const std::sting prerelax = "A",
				const std::sting prerelax = "FFFC",
    			const int relax_type = 0,
    			const int interp_type = 100,
    			const int coarsen_type = 6,
    			const int print_level = 3,
    			const int max_itter = 1000,
    			const int max_levels = 50,
    			const int cycle_type = 1,
    			const int debug_flag = 0,
    			const int sabs_flag = 0,
    			const int trilinos_print_time = 1,
    			const int amg_logging = 0,
    			const double strength_tolC = 5.0e-3,
    			const double strength_tolR = 5.0e-3,
    			const double distance_R = 2.0,
    			const double filterA_tol = 1.0e-4,
    			const double solve_tol = 1e-10,
    			const double post_filter_R = 0.0

    	);

        std::string prerelax;
        std::string postrelax;
        int relax_type;
        int interp_type;
        int coarsen_type;
        int print_level;
        int max_itter;
        int max_levels;
        int cycle_type;
        int debug_flag;
        int sabs_flag;
        int trilinos_print_time;
        int amg_logging;
        double strength_tolC;
        double strength_tolR;
        double distance_R;
        double filterA_tol;
        double solve_tol;
        double post_filter_R;
    };
    /**
     * Constructor. Takes the solver control object and creates the solver.
     */
    SolverBoomerAMG(SolverControl &       cn,
                 const AdditionalData &data = AdditionalData());

};

}

DEAL_II_NAMESPACE_CLOSE
