#include<Ifpack_Hypre.h>
#include<Epetra_MultiVector.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/base/config.h>

DEAL_II_NAMESPACE_OPEN

namespace TrilinosWrappers {
/**
 * An implementation of the hypre BoomerAMG solver accessed
 * through the Trilinos package ifpack.
 *
 * @ingroup TrilinosWrappers
 * @author Joshua Hanophy, 2019
 */
class SolverBoomerAMG {
public:
	/**
	 * This struct
	 */
	struct AdditionalData {
		explicit AdditionalData(const std::sting prerelax = "A",
				const std::sting prerelax = "FFFC", const int relax_type = 0,
				const int interp_type = 100, const int coarsen_type = 6,
				const int print_level = 3, const int max_iteration = 1000,
				const int max_levels = 50, const int cycle_type = 1,
				const int debug_flag = 0, const int sabs_flag = 0,
				const int trilinos_print_time = 1, const int amg_logging = 0,
				const double strength_tolC = 5.0e-3,
				const double strength_tolR = 5.0e-3, const double distance_R =
						2.0, const double filterA_tol = 1.0e-4,
				const double solve_tol = 1e-10, const double post_filter_R = 0.0

				);
		/**
		 * The prerelax string specifies the points, order, and relaxation steps
		 * for prerelaxation. The options of "A", "F", or "C" where A is relaxation over
		 * all points, F is relaxation over the F-points, and C is relaxation over the
		 * C-points. Multiple characters specify multiple relaxation steps and the order
		 * matters. For example, "AA" specifies two relaxation steps of all points.
		 */
		std::string prerelax;
		/**
		 * The postrelax string specifies the points, order, and relaxation steps
		 * for postrelaxation. The options of "A", "F", or "C" where A is relaxation over
		 * all points, F is relaxation over the F-points, and C is relaxation over the
		 * C-points. Multiple characters specify multiple relaxation steps and the order
		 * matters. For example, "FFFC" specifies three post relaxations over F-points
		 * followed by a relexation over C-points.
		 */
		std::string postrelax;
		/**
		 * The relax_type integer variable sets the relaxation type.
		 * Relaxation types, taken from the hypre documentation, are:
		 * <ul>
		 * <li> 0: Jacobi </li>
		 * <li> 1: Gauss-Seidel, sequential (very slow!) </li>
		 * <li> 2: Gauss-Seidel, interior points in parallel, boundary sequential (slow!) </li>
		 * <li> 3: hybrid Gauss-Seidel or SOR, forward solve </li>
		 * <li> 4: hybrid Gauss-Seidel or SOR, backward solve </li>
		 * <li> 5: hybrid chaotic Gauss-Seidel (works only with OpenMP) </li>
		 * <li> 6: hybrid symmetric Gauss-Seidel or SSOR </li>
		 * <li> 8: \f$\ell_1\f$ Gauss-Seidel, forward solve </li>
		 * <li> 9: Gaussian elimination (only on coarsest level) </li>
		 * <li> 13: \f$\ell_1\f$ Gauss-Seidel, forward solve </li>
		 * <li> 14: \f$\ell_1\f$ Gauss-Seidel, backward solve </li>
		 * <li> 15: CG (warning - not a fixed smoother - may require FGMRES) </li>
		 * <li> 16: Chebyshev </li>
		 * <li> 17: FCF-Jacobi </li>
		 * <li> 18: \f$\ell_1\f$-scaled jacobi </li>
		 * </ul>
		 */
		int relax_type;
		/**
		 * The interp_type integer variable sets the interpolation type.
		 * Interpolation types, taken from the hypre documentation, are:
		 * <ul>
		 * <li> 0: classical modified interpolation </li>
		 * <li> 1: LS interpolation (for use with GSMG) </li>
		 * <li> 2: classical modified interpolation for hyperbolic PDEs </li>
		 * <li> 3: direct interpolation (with separation of weights) </li>
		 * <li> 4: multipass interpolation </li>
		 * <li> 5: multipass interpolation (with separation of weights) </li>
		 * <li> 7: extended+i interpolation </li>
		 * <li> 8: standard interpolation </li>
		 * <li> 9: standard interpolation (with separation of weights) </li>
		 * <li> 10: classical block interpolation (for use with nodal systems version only) </li>
		 * <li> 11: classical block interpolation (for use with nodal systems version only)<br/>
		 * with diagonalized diagonal blocks<br/></li>
		 * <li> 12: FF interpolation </li>
		 * <li> 13: FF1 interpolation </li>
		 * <li> 14: extended interpolation </li>
		 * <li> 100: Pointwise interpolation (intended for use with AIR) </li>
		 * </ul>
		 */
		int interp_type;
		/**
		 * The coarsen_type integer variable sets the coarsening algorithm.
		 * Coarsening algorithm options, taken from the hypre documentation, are:
		 * <ul>
		 * <li> 0: CLJP-coarsening (a parallel coarsening algorithm using independent sets. </li>
		 * <li> 3: classical Ruge-Stueben coarsening on each processor, followed by a third pass,<br/>
		 * which adds coarse points on the boundaries <br/></li>
		 * <li> 6: Falgout coarsening (uses 1 first, followed by CLJP using the interior coarse points<br/>
		 * generated by 1 as its first independent set) <br/></li>
		 * <li> 8: PMIS-coarsening (a parallel coarsening algorithm using independent sets, generating<br/>
		 * lower complexities than CLJP, might also lead to slower convergence) <br/></li>
		 * <li> 10: HMIS-coarsening (uses one pass Ruge-Stueben on each processor independently, followed<br/>
		 *  by PMIS using the interior C-points generated as its first independent set) <br/></li>
		 * <li> 21: CGC coarsening by M. Griebel, B. Metsch and A. Schweitzer </li>
		 * <li> 22: CGC-E coarsening by M. Griebel, B. Metsch and A.Schweitzer </li>
		 * </ul>
		 */
		int coarsen_type;
		/**
		 *
		 */
		int print_level;
		/**
		 *
		 */
		int max_itter;
		/**
		 *
		 */
		int max_levels;
		/**
		 * The cycle_type integer variable sets the cycle type. Cycle types available,
		 * taken from the hypre documentation, are:
		 * <ul>
		 * <li> 0: F-cycle type </li>
		 * <li> 1: V-cycle type </li>
		 * <li> 2: W-cycle type </li>
		 * </ul>
		 */
		int cycle_type;
		/**
		 *
		 */
		int debug_flag;
		/**
		 * sabs_flag sets whether the classical strength of connection test
		 * based on testing the negative of matrix ocefficient or if the absolute
		 * value is tested. If set to 0, the negative coefficient values are tested,
		 * if set to 1, the absolute values of matrix coefficients are tested.
		 */
		int sabs_flag;
		/**
		 *
		 */
		int trilinos_print_time;
		/**
		 *
		 */
		int amg_logging;
		/**
		 *
		 */
		double strength_tolC;
		/**
		 *
		 */
		double strength_tolR;
		/**
		 *
		 */
		double distance_R;
		/**
		 *
		 */
		double filterA_tol;
		/**
		 *
		 */
		double solve_tol;
		/**
		 *
		 */
		double post_filter_R;
	};
	/**
	 * Constructor. Takes the solver control object and creates the solver.
	 */
	SolverBoomerAMG(SolverControl & cn, const AdditionalData &data =
			AdditionalData());

};

}

DEAL_II_NAMESPACE_CLOSE
