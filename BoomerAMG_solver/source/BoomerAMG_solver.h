#include<Ifpack_Hypre.h>
#include<Epetra_MultiVector.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>

#include "boost/variant.hpp"

DEAL_II_NAMESPACE_OPEN

namespace TrilinosWrappers {

namespace LA =  dealii::LinearAlgebraTrilinos::MPI;

class ifpackHypreSolverPrecondParameters{
public:
	/**
	 * This type is meant to be used for a pointer to a hypre set function. This is simply an alias for a
	 * boost variant type. Note the types in the boos variant contianer are those function pointer types
	 * supported by this interface.
	 */
	typedef boost::variant<int (*)(HYPRE_Solver, int),int (*)(HYPRE_Solver, double),int (*)(HYPRE_Solver,double, int),
			int (*)(HYPRE_Solver, int, int),int (*)(HYPRE_Solver, int*),int (*)(HYPRE_Solver, double*),
			int (*)(HYPRE_Solver, int**),std::nullptr_t> hypre_function_variant;
	/**
	 * This type is meant to be used for a parameter value. This is simply an alias for a boost variant type.
	 */
	typedef boost::variant<int,double,int*,int**,std::pair<double,int>, std::pair<int, int>,
            std::pair<std::string,std::string>> param_value_variant;
	/**
	 * This struct holds the data required for a single parameter. The required data is a parameter value and a set function.
	 * There are two possibilities for a set function. A pointer to a hypre set function defined in the hypre library can
	 * used and will be stored as hypre_function. Or if a custom set function is desired in place of directly using the predfined
	 * hypre set functions, a pointer to a custom set function is stored as set_function. Note that either set_function or
	 * hypre_function shoudl be a nullptr.
	 */
	struct parameter_data{
		/**
		 * value stores the value of the parameter
		 */
		param_value_variant value;
		/**
		 * hypre_function stores a pointer to the hypre set function to be used to set the parameter value. Note is this is used
		 * then hypre_function should be equal to a nullptr
		 */
		hypre_function_variant hypre_function=nullptr;
		/**
		 * set_function stores a pointer to a custom set function. This can be used if simply setting a parameter value with the
		 * set functions predifined in the hypre library is not sufficient. If this is used, hypre_function should be equal to nullptr
		 */
		std::function<void(const Hypre_Chooser, const parameter_data &, Ifpack_Hypre &)> set_function=nullptr;
		/**
		 * Constructor.
		 *
		 * @param value is the value of the parameter
		 * @param hypre_function is a pointer to the hypre set function defined in the hypre library
		 */
		parameter_data(param_value_variant value, hypre_function_variant hypre_function):value(value),hypre_function(hypre_function){};
		/**
		 * Constructor.
		 *
		 * @param value is the value of the parameter
		 * @param set_function is a pointer to a custom set function
		 */
		parameter_data(param_value_variant value, std::function<void(const Hypre_Chooser, const parameter_data &, Ifpack_Hypre &)> set_function):value(value),set_function(set_function){};
	};


	ifpackHypreSolverPrecondParameters(Hypre_Chooser solver_preconditioner_selection):solver_preconditioner_selection(solver_preconditioner_selection){};

	/**
	 * This function can be used to change the value of a parameter in the parameters map that
	 * already exists. Use the add_parameter function if the parameter does not already exist
	 *
	 * @param name is the string parameter name. Note that the parameter name should already exist
	 * in the parameters parameter map. Use the add_parameter function to add a new parameters.
	 * @param value This is the value to assign the parameter found at parameters[name]
	 */
	void set_parameter_value(std::string name, param_value_variant value);
	/**
	 *
	 *
	 * @param name is the string parameter name. Note that the parameter name should not already
	 * exist. To update the value or a parameter that already exists, use the set_parameter_value
	 * function
	 * @param param_data is an instance of the parameter_data struct which contains the parameter
	 * value and a pointer to the proper set function
	 */
	void add_parameter(std::string name, parameter_data param_data);

	/**
	 * Function to remove a parameter from the parameters parameter map
	 *
	 * @param name is the string parameter name to remove.
	 */
	void remove_parameter(std::string name);

	/**
	 * Function to return the value of a parameter.
	 *
	 * @param name is the string parameter name of the parameter whose value is to be returned
	 */
	template<typename return_type>
	void return_parameter_value(std::string name);
	/**
	 * This function is to be used by the solver or preconditioner class to set
	 */
	void set_parameters(Ifpack_Hypre & Ifpack_obj);
	/**
	 * This is a special set function used to simplify the specification of relaxation orders when using
	 * AIR amg.
	 */
	static void set_relaxation_order(const Hypre_Chooser solver_preconditioner_selection, const parameter_data & param_data, Ifpack_Hypre & Ifpack_obj);

private:

	const Hypre_Chooser solver_preconditioner_selection;

	/**
	 * The parameters map stores parameters as a string name key and then a parameter_data instance value.
	 */
	std::map< std::string,parameter_data> parameters;
	/**
	 * This class is used internally to set parameter values
	 */
	class apply_parameter_variant_visitor:
			public boost::static_visitor<>
	{
	public:
		apply_parameter_variant_visitor(Ifpack_Hypre & Ifpack_obj, const Hypre_Chooser solver_preconditioner_selection )
		:Ifpack_obj(Ifpack_obj),solver_preconditioner_selection(solver_preconditioner_selection){};

		void operator()( int (* hypre_set_func)(HYPRE_Solver, int) & , int & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value);
		}

		void operator()( int (* hypre_set_func)(HYPRE_Solver, double) & , double & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value);
		}

		void operator()( int (* hypre_set_func)(HYPRE_Solver, double, int) & , std::pair<double,int> & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value.first,value.second);
		}

		void operator()( int (* hypre_set_func)(HYPRE_Solver, int, int) & , std::pair<int,int> & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value.first,value.second);
		}

		void operator()( int (* hypre_set_func)(HYPRE_Solver, int*) & , int* & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value);
		}

		void operator()( int (* hypre_set_func)(HYPRE_Solver, double*) & , double* & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value);
		}

		void operator()( int (* hypre_set_func)(HYPRE_Solver, int**) & , int** & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value);
		}

		template <typename T, typename U>
		void operator()(T & func, U & value){
			(void) func;
			(void) value;

			AssertThrow(false, ExcMessage("When setting a parameter, the hypre set function prototype\nshould match the type of the parameter value given"));
		}

	private:
		Ifpack_Hypre & Ifpack_obj;
		const Hypre_Chooser solver_preconditioner_selection;
	};
	/**
	 * This class is used internally to return parameter values
	 */
	class return_value_visitor:
			public boost::static_visitor<>
	{
	public:
		template<typename T>
		void operator()(T& value){
			return value;
		}
	};


};


/**
 * The possible parameters are: <table> <tr>
 * <td align="center"> interp_type </td>
 * <td align="left">
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
 * </ul> </td> </tr> <tr>
 *
 * <td align="center"> pre_post_relax </td>
 * <td align="left">
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
 * </td></tr> <tr>
 * <td align="center"> relax_type </td>
 * <td align="left">
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
 * </td></tr> <tr>
 *
 * <td align="center"> coarsen_type </td>
 * <td align="left">
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
 * </td></tr> <tr>
 *
 * <td align="center"> max_levels </td>
 * <td align="left">
 * The max_levels integer specifies the maximum number of AMG that hypre
 * will be allowed to use
 * </td></tr> <tr>
 *
 * <td align="center"> cycle_type </td>
 * <td align="left">
 * The cycle_type integer variable sets the cycle type. Cycle types available,
 * taken from the hypre documentation, are:
 * <ul>
 * <li> 0: F-cycle type </li>
 * <li> 1: V-cycle type </li>
 * <li> 2: W-cycle type </li>
 * </ul>
 * </td></tr> <tr>
 *
 * <td align="center"> sabs_flag </td>
 * <td align="left">
 * sabs_flag sets whether the classical strength of connection test
 * based on testing the negative of matrix coefficient or if the absolute
 * value is tested. If set to 0, the negative coefficient values are tested,
 * if set to 1, the absolute values of matrix coefficients are tested.
 * </td></tr> <tr>
 *
 * <td align="center"> distance_R </td>
 * <td align="left">
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
 * </td></tr> <tr>
 * </table>
 *
 * @ingroup TrilinosWrappers
 * @author Joshua Hanophy, Ben Southworth 2019
 */
class BoomerAMG_Parameters{
friend class SolverBoomerAMG;
friend class BoomerAMG_PreconditionedSolver;
public:

	/**
	 * The configuratoin_types enum is used to select a default variable values when
	 * constructing the BoomerAMG_Parameters object
	 */
	enum default_configuration_type {
		/**
		 * Load default parameters consistent with Classical AMG
		 */
		CLASSICAL_AMG,
		/**
		 * Load default parameters consistent with AIR
		 */
		AIR_AMG,
		/**
		 * Create an empty parameter map
		 */
		NONE
	};
	/**
	 * This type is meant to be used for a pointer to a hypre set function. This is simply an alias for a
	 * boost variant type. Note the types in the boos variant contianer are those function pointer types
	 * supported by this interface.
	 */
	typedef boost::variant<int (*)(HYPRE_Solver, int),int (*)(HYPRE_Solver, double),int (*)(HYPRE_Solver,double, int),
			int (*)(HYPRE_Solver, int, int),int (*)(HYPRE_Solver, int*),int (*)(HYPRE_Solver, double*),
			int (*)(HYPRE_Solver, int**),std::nullptr_t> hypre_function_variant;
	/**
	 * This type is meant to be used for a parameter value. This is simply an alias for a boost variant type.
	 */
	typedef boost::variant<int,double,int*,int**,std::pair<double,int>, std::pair<int, int>,
            std::pair<std::string,std::string>> param_value_variant;
	/**
	 * This struct holds the data required for a single parameter. The required data is a parameter value and a set function.
	 * There are two possibilities for a set function. A pointer to a hypre set function defined in the hypre library can
	 * used and will be stored as hypre_function. Or if a custom set function is desired in place of directly using the predfined
	 * hypre set functions, a pointer to a custom set function is stored as set_function. Note that either set_function or
	 * hypre_function shoudl be a nullptr.
	 */
	struct parameter_data{
		/**
		 * value stores the value of the parameter
		 */
		param_value_variant value;
		/**
		 * hypre_function stores a pointer to the hypre set function to be used to set the parameter value. Note is this is used
		 * then hypre_function should be equal to a nullptr
		 */
		hypre_function_variant hypre_function=nullptr;
		/**
		 * set_function stores a pointer to a custom set function. This can be used if simply setting a parameter value with the
		 * set functions predifined in the hypre library is not sufficient. If this is used, hypre_function should be equal to nullptr
		 */
		std::function<void(const Hypre_Chooser, const parameter_data &, Ifpack_Hypre &)> set_function=nullptr;
		/**
		 * Constructor.
		 *
		 * @param value is the value of the parameter
		 * @param hypre_function is a pointer to the hypre set function defined in the hypre library
		 */
		parameter_data(param_value_variant value, hypre_function_variant hypre_function):value(value),hypre_function(hypre_function){};
		/**
		 * Constructor.
		 *
		 * @param value is the value of the parameter
		 * @param set_function is a pointer to a custom set function
		 */
		parameter_data(param_value_variant value, std::function<void(const Hypre_Chooser, const parameter_data &, Ifpack_Hypre &)> set_function):value(value),set_function(set_function){};
	};
	/**
	 * Constructor.
	 *
	 * @param config_selection specifies which type of defualt configuration will be used.
	 * The default parameter configuration dictates which parameters are used and their values.
	 * Note that default_configuration_type::NONE can be specified which result in the construction
	 * of a an empty parameters map which could then be filled with the add_parameter function.
	 */
	BoomerAMG_Parameters(default_configuration_type config_selection);
	/**
	 * This function can be used to change the value of a parameter in the parameters map that
	 * already exists. Use the add_parameter function if the parameter does not already exist
	 *
	 * @param name is the string parameter name. Note that the parameter name should already exist
	 * in the parameters parameter map. Use the add_parameter function to add a new parameters.
	 * @param value This is the value to assign the parameter found at parameters[name]
	 */
	void set_parameter_value(std::string name, param_value_variant value);
	/**
	 *
	 *
	 * @param name is the string parameter name. Note that the parameter name should not already
	 * exist. To update the value or a parameter that already exists, use the set_parameter_value
	 * function
	 * @param param_data is an instance of the parameter_data struct which contains the parameter
	 * value and a pointer to the proper set function
	 */
	void add_parameter(std::string name, parameter_data param_data);

	/**
	 * Function to remove a parameter from the parameters parameter map
	 *
	 * @param name is the string parameter name to remove.
	 */
	void remove_parameter(std::string name);

	/**
	 * Function to return the value of a parameter.
	 *
	 * @param name is the string parameter name of the parameter whose value is to be returned
	 */
	template<typename return_type>
	void return_parameter_value(std::string name);

private:
	/**
	 * The parameters map stores parameters as a string name key and then a parameter_data instance value.
	 */
	std::map< std::string,parameter_data> parameters;
	/**
	 * This is a special set function used to simplify the specification of relaxation orders when using
	 * AIR amg.
	 */
	static void set_relaxation_order(const Hypre_Chooser solver_preconditioner_selection, const parameter_data & param_data, Ifpack_Hypre & Ifpack_obj);
	/**
	 * This function is to be used by the solver or preconditioner class to set
	 */
	void set_parameters(Ifpack_Hypre & Ifpack_obj, const Hypre_Chooser solver_preconditioner_selection);
	/**
	 * This class is used internally to set parameter values
	 */
	class apply_parameter_variant_visitor:
			public boost::static_visitor<>
	{
	public:
		apply_parameter_variant_visitor(Ifpack_Hypre & Ifpack_obj, const Hypre_Chooser solver_preconditioner_selection )
		:Ifpack_obj(Ifpack_obj),solver_preconditioner_selection(solver_preconditioner_selection){};

		void operator()( int (* hypre_set_func)(HYPRE_Solver, int) & , int & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value);
		}

		void operator()( int (* hypre_set_func)(HYPRE_Solver, double) & , double & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value);
		}

		void operator()( int (* hypre_set_func)(HYPRE_Solver, double, int) & , std::pair<double,int> & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value.first,value.second);
		}

		void operator()( int (* hypre_set_func)(HYPRE_Solver, int, int) & , std::pair<int,int> & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value.first,value.second);
		}

		void operator()( int (* hypre_set_func)(HYPRE_Solver, int*) & , int* & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value);
		}

		void operator()( int (* hypre_set_func)(HYPRE_Solver, double*) & , double* & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value);
		}

		void operator()( int (* hypre_set_func)(HYPRE_Solver, int**) & , int** & value){
			Ifpack_obj.SetParameter(solver_preconditioner_selection,hypre_set_func,value);
		}

		template <typename T, typename U>
		void operator()(T & func, U & value){
			(void) func;
			(void) value;

			AssertThrow(false, ExcMessage("When setting a parameter, the hypre set function prototype\nshould match the type of the parameter value given"));
		}

	private:
		Ifpack_Hypre & Ifpack_obj;
		const Hypre_Chooser solver_preconditioner_selection;
	};
	/**
	 * This class is used internally to return parameter values
	 */
	class return_value_visitor:
			public boost::static_visitor<>
	{
	public:
		template<typename T>
		void operator()(T& value){
			return value;
		}
	};

};

class SolverBoomerAMG{
public:

	enum AMG_type {
		/**
		 * Load default parameters consistent with Classical AMG
		 */
		CLASSICAL_AMG,
		/**
		 * Load default parameters consistent with AIR
		 */
		AIR_AMG,
		/**
		 * Create an empty parameter map
		 */
		NONE
	};

	/**
	 * Constructor.
	 *
	 * @param parameters_obj
	 */

	SolverBoomerAMG(AMG_type config_selection);
	/**
	 * Solver function
	 *
	 * @param system_matrix
	 * @param right_hand_side
	 * @param solution
	 */
	void solve(LA::SparseMatrix & system_matrix,
			   LA::Vector & right_hand_side,
			   LA::Vector &solution);

	ifpackHypreSolverPrecondParameters SolverParameters;

};

class BoomerAMG_PreconditionedSolver{
public:

	/**
	 * Constructor.
	 *
	 * @param parameters_obj
	 */

    struct AdditionalData
    {
    	// Solver params can go here
    };

    BoomerAMG_PreconditionedSolver(BoomerAMG_Parameters & parameters_obj, Hypre_Solver solver_selection=Hypre_Solver::PCG);

	void solve(LA::SparseMatrix & system_matrix,
			   LA::Vector & right_hand_side,
			   LA::Vector &solution);

private:
	BoomerAMG_Parameters & parameters_obj;

	Hypre_Solver solver_selection;

};


} // Close namespace TrilinosWrappers
DEAL_II_NAMESPACE_CLOSE
