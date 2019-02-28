#include<Ifpack_Hypre.h>
#include<Epetra_MultiVector.h>

#include <deal.II/lac/trilinos_vector.h>
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
class BoomerAMG_Parameters{
public:
	Hypre_Chooser solver_preconditioner_selection;
	/**
	 * BoomerAMG_Parameter_Base is a common base class for all parameters to inherit from.
	 * The primary purpose of this class is to have a single pointer type for each parameter
	 */
	class BoomerAMG_Parameter_Base{
		friend class BoomerAMG_Parameters;
	protected:
		static std::vector<BoomerAMG_Parameter_Base *> parameter_list;
		bool parameter_used=true;
		virtual void set_parameter(Hypre_Chooser, std::vector<FunctionParameter> *) = 0;
	};
	/**
	 * The BoomerAMG_Parameter abstract base class specifies how a parameter
	 * should look.
	 */
	template<class parameter_type>
	class BoomerAMG_Parameter :public BoomerAMG_Parameter_Base{
	public:
		BoomerAMG_Parameter();
		parameter_type value;
	};

	/**
	 * The prerelax string specifies the points, order, and relaxation steps
	 * for prerelaxation. The options are "A", "F", or "C" where A is relaxation over
	 * all points, F is relaxation over the F-points, and C is relaxation over the
	 * C-points. Multiple characters specify multiple relaxation steps and the order
	 * matters. For example, "AA" specifies two relaxation steps of all points.
	 *
	 * The postrelax string specifies the points, order, and relaxation steps
	 * for postrelaxation. The options are "A", "F", or "C" where A is relaxation over
	 * all points, F is relaxation over the F-points, and C is relaxation over the
	 * C-points. Multiple characters specify multiple relaxation steps and the order
	 * matters. For example, "FFFC" specifies three post relaxations over F-points
	 * followed by a relexation over C-points.
	 */
	class pre_post_relax: public BoomerAMG_Parameter<std::pair<std::string,std::string>>{
		//pre_post_relax();
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter> *);
	};

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
	class relax_type : public BoomerAMG_Parameter <int>{
		//relax_type();
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter> *);
	};
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
	class interp_type : public BoomerAMG_Parameter <int>{
		//interp_type();
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter> *);
	};
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
	class coarsen_type : public BoomerAMG_Parameter <int>{
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter> *);
	};

	class print_level : public BoomerAMG_Parameter <int>{
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter > *);
	};

	//int max_itter;

	/**
	 * The max_levels integer specifies the maximum number of AMG that hypre
	 * will be allowed to use
	 */
	class max_levels : public BoomerAMG_Parameter <int>{
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter > *);
	};

	/**
	 * The cycle_type integer variable sets the cycle type. Cycle types available,
	 * taken from the hypre documentation, are:
	 * <ul>
	 * <li> 0: F-cycle type </li>
	 * <li> 1: V-cycle type </li>
	 * <li> 2: W-cycle type </li>
	 * </ul>
	 */
	class cycle_type : public BoomerAMG_Parameter <int>{
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter > *);
	};

	/**
	 */
	class debug_flag : public BoomerAMG_Parameter <int>{
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter > *);
	};
	/**
	 * sabs_flag sets whether the classical strength of connection test
	 * based on testing the negative of matrix coefficient or if the absolute
	 * value is tested. If set to 0, the negative coefficient values are tested,
	 * if set to 1, the absolute values of matrix coefficients are tested.
	 */
	class sabs_flag : public BoomerAMG_Parameter <int>{
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter> *);
	};

	class print_ifpack_timing : public BoomerAMG_Parameter <int>{
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter> *);
	};

	class strength_tolC : public BoomerAMG_Parameter <double>{
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter> *);
	};

	class strength_tolR : public BoomerAMG_Parameter <double>{
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter> *);
	};

	/**
	 * The distance_R double variable sets whether Approximate Ideal Restriction
	 * (AIR) multigrid or classical multigrid is used.
	 * <ul>
	 * <li> 0.0: Use classical AMG, not AIR </li>
	 * <li> 1.0: Use AIR, Distance-1 LAIR is used to compute R </li>
	 * <li> 2.0: Use AIR, Distance-2 LAIR is used to compute R </li>
	 * <li> 3.0: Use AIR, degree 0 Neumann expansion is used to compute R </li>
	 * <li> 4.0: Use AIR, degree 1 Neumann expansion is used to compute R </li>
	 * <li> 5.0: Use AIR, degree 2 Neumann expansion is used to compute R </li>
	 * </ul>
	 */
	class distance_R : public BoomerAMG_Parameter <double>{
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter> *);
	};

	class filterA_tol : public BoomerAMG_Parameter <double>{
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter> *);
	};

	class solve_tol : public BoomerAMG_Parameter <double>{
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter> *);
	};

	class post_filter_R : public BoomerAMG_Parameter <double>{
		void set_parameter(Hypre_Chooser, std::vector<FunctionParameter> *);
	};
	/**
	 * The configuratoin_types enum is used to select a default variable values when
	 * constructing the BoomerAMG_Parameters object
	 */
	enum configuration_types {CLASSICAL_AMG,AIR,NONE};


	BoomerAMG_Parameters(BoomerAMG_Parameters::configuration_types config_selection);

	void set_parameter_list(std::vector<FunctionParameter> *);

};

template<class parameter_type>
BoomerAMG_Parameters::BoomerAMG_Parameter<parameter_type>::BoomerAMG_Parameter(){
	parameter_list.push_back(this);
}



} // Close namespace TrilinosWrappers
DEAL_II_NAMESPACE_CLOSE
