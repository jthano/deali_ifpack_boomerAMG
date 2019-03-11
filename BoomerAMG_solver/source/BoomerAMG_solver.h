#include<Ifpack_Hypre.h>
#include<Epetra_MultiVector.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/base/config.h>

#include "boost/variant.hpp"

DEAL_II_NAMESPACE_OPEN

namespace TrilinosWrappers {

namespace LA =  dealii::LinearAlgebraTrilinos::MPI;

class BoomerAMG_Parameters{

public:

	typedef boost::variant<int (*)(HYPRE_Solver, int),int (*)(HYPRE_Solver, double),int (*)(HYPRE_Solver,double, int),
			int (*)(HYPRE_Solver, int, int),int (*)(HYPRE_Solver, int*),int (*)(HYPRE_Solver, double*),
			int (*)(HYPRE_Solver, int**),std::nullptr_t> hypre_function_variant;

	typedef boost::variant<int,double,int*,int**,std::pair<double,int>, std::pair<int, int>,
            std::pair<std::string,std::string>> param_value_variant;

	struct parameter_data{
		param_value_variant value;
		hypre_function_variant hypre_function=nullptr;
		std::function<void(const Hypre_Chooser, const parameter_data &, Ifpack_Hypre &)> set_function=nullptr;
		parameter_data(param_value_variant value, hypre_function_variant hypre_function):value(value),hypre_function(hypre_function){};
	};

	BoomerAMG_Parameters();

	enum default_configuration_type{AIR_AMG,CLASSICAL_AMG,NONE};

	void set_parameters(Ifpack_Hypre & Ifpack_obj, const Hypre_Chooser solver_preconditioner_selection);

	std::map< std::string,parameter_data> parameters;

private:

	void set_classical_amg_parameters();
	void set_air_amg_parameters();

	void set_relaxation_order(const Hypre_Chooser solver_preconditioner_selection, const parameter_data & param_data, Ifpack_Hypre & Ifpack_obj);

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
			// It will be an error to have any other combination of items
			(void) func;
			(void) value;
		}

	private:
		Ifpack_Hypre & Ifpack_obj;
		const Hypre_Chooser solver_preconditioner_selection;
	};

};

class SolverBoomerAMG{
public:

	SolverBoomerAMG(BoomerAMG_Parameters & parameters_obj):parameters_obj(parameters_obj){};

	void solve(LA::SparseMatrix & system_matrix,
			   LA::Vector & right_hand_side,
			   LA::Vector &solution);

private:
	BoomerAMG_Parameters & parameters_obj;

};

} // Close namespace TrilinosWrappers
DEAL_II_NAMESPACE_CLOSE
