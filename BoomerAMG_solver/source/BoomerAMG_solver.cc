
#include <BoomerAMG_solver.h>

DEAL_II_NAMESPACE_OPEN

namespace TrilinosWrappers
{


void ifpackHypreSolverPrecondParameters::set_parameters(Ifpack_Hypre & Ifpack_obj){

	apply_parameter_variant_visitor parameter_visitor(Ifpack_obj, solver_preconditioner_selection);

	for (auto param_itter=parameters.begin();param_itter!=parameters.end();++param_itter){
		if ((param_itter->second).set_function == nullptr){
			boost::apply_visitor(parameter_visitor, (param_itter->second).hypre_function, (param_itter->second).value );
		} else{
			(param_itter->second).set_function(solver_preconditioner_selection, param_itter->second , Ifpack_obj);
		}
	}
}


void ifpackHypreSolverPrecondParameters::set_parameter_value(const std::string name,const param_value_variant value){

	auto it = parameters.find(name);

	AssertThrow(it!=parameters.end(), ExcMessage("When using set_parameter_value, the parameter must already be present in the parameters map."));

	(it->second).value = value;
}

void ifpackHypreSolverPrecondParameters::add_parameter(const std::string name,const parameter_data param_data){

	auto it = parameters.find(name);

	AssertThrow(it==parameters.end(), ExcMessage("When using add_parameter, the parameter name should not already exist."));

	parameters.insert({name, param_data});

}

void ifpackHypreSolverPrecondParameters::remove_parameter(const std::string name){

	auto it = parameters.find(name);

	if (it!=parameters.end())
		parameters.erase(it);

}
/**
 * TODO:: Make this const function because should not modify anything, only return value
 * @param name
 */
template<typename return_type>
void ifpackHypreSolverPrecondParameters::return_parameter_value(std::string name){

	return_value_visitor value_visitor;

	return boost::apply_visitor(value_visitor, parameters[name].value);

}


ifpackSolverParameters::ifpackSolverParameters(Hypre_Solver solver_selection/*=Hypre_Solver::PCG*/)
:
solver_selection(solver_selection),
ifpackHypreSolverPrecondParameters(Hypre_Chooser::Solver)
{

}

void BoomerAMGParameters::set_relaxation_order(const Hypre_Chooser solver_preconditioner_selection, const parameter_data & param_data, Ifpack_Hypre & Ifpack_obj){

	std::pair<std::string,std::string> param_value = boost::get< std::pair<std::string,std::string> >(param_data.value);

	const unsigned int ns_down = param_value.first.length();
	const unsigned int ns_up = param_value.second.length();
	const unsigned int ns_coarse = 1 ;

	const std::string F("F");
	const std::string C("C");
	const std::string A("A");

	// Array to store relaxation scheme and pass to Hypre
	int **grid_relax_points = (int **) malloc(4*sizeof(int *));
	grid_relax_points[0] = NULL;
	grid_relax_points[1] = std::unique_ptr<int>(new int [ns_down]); //(int *) malloc(sizeof(int)*ns_down);
	grid_relax_points[2] = std::unique_ptr<int>(new int [ns_up]);//(int *) malloc(sizeof(int)*ns_up);
	grid_relax_points[3] = std::unique_ptr<int>(new int [0]);//(int *) malloc(sizeof(int));
	grid_relax_points[3][0] = 0;

	// set down relax scheme
	for(unsigned int i = 0; i<ns_down; i++) {
	    if (param_value.first.compare(i,1,F) == 0) {
	       grid_relax_points[1][i] = -1;
	    }
	    else if (param_value.first.compare(i,1,C) == 0) {
	       grid_relax_points[1][i] = 1;
	    }
	    else if (param_value.first.compare(i,1,A) == 0) {
	       grid_relax_points[1][i] = 0;
	    }
	 }

	 // set up relax scheme
	 for(unsigned int i = 0; i<ns_up; i++) {
	    if (param_value.second.compare(i,1,F) == 0) {
	       grid_relax_points[2][i] = -1;
	    }
	    else if (param_value.second.compare(i,1,C) == 0) {
	       grid_relax_points[2][i] = 1;
	    }
	    else if (param_value.second.compare(i,1,A) == 0) {
	       grid_relax_points[2][i] = 0;
	    }
	 }

	Ifpack_obj.SetParameter(solver_preconditioner_selection , & HYPRE_BoomerAMGSetGridRelaxPoints , grid_relax_points);
	Ifpack_obj.SetParameter(solver_preconditioner_selection , & HYPRE_BoomerAMGSetCycleNumSweeps , ns_coarse,3);
	Ifpack_obj.SetParameter(solver_preconditioner_selection , & HYPRE_BoomerAMGSetCycleNumSweeps , ns_down,1);
	Ifpack_obj.SetParameter(solver_preconditioner_selection , & HYPRE_BoomerAMGSetCycleNumSweeps , ns_up,2);

}


BoomerAMGParameters::BoomerAMGParameters(const AMG_type config_selection)
:ifpackHypreSolverPrecondParameters(Hypre_Chooser::Preconditioner)
{
	set_common_AMG_parameters(config_selection);
	parameters.insert( {"hypre_print_level", parameter_data(1, & HYPRE_BoomerAMGSetPrintLevel)} );
	parameters.insert( {"solve_tol", parameter_data(0.0, & HYPRE_BoomerAMGSetTol)} );
	parameters.insert( {"max_itter", parameter_data(1, & HYPRE_BoomerAMGSetMaxIter)} );
}

BoomerAMGParameters::BoomerAMGParameters(const int max_itter,const double solv_tol,const AMG_type config_selection)
:ifpackHypreSolverPrecondParameters(Hypre_Chooser::Solver)
{
	set_common_AMG_parameters(config_selection);
	parameters.insert( {"hypre_print_level", parameter_data(3, & HYPRE_BoomerAMGSetPrintLevel)} );
	parameters.insert( {"solve_tol", parameter_data(solv_tol, & HYPRE_BoomerAMGSetTol)} );
	parameters.insert( {"max_itter", parameter_data(max_itter, & HYPRE_BoomerAMGSetMaxIter)} );
}

void BoomerAMGParameters::set_common_AMG_parameters(const AMG_type config_selection){
	switch(config_selection)
	{
	case AIR_AMG:
	{
		parameters.insert( {"interp_type", parameter_data(100, & HYPRE_BoomerAMGSetInterpType)} );
		parameters.insert( {"coarsen_type", parameter_data(6, & HYPRE_BoomerAMGSetCoarsenType)} );
		parameters.insert( {"relax_type", parameter_data(0, & HYPRE_BoomerAMGSetRelaxType)} );
		parameters.insert( {"max_amg_levels", parameter_data(40, & HYPRE_BoomerAMGSetMaxLevels)} );
		parameters.insert( {"sabs_flag", parameter_data(1, & HYPRE_BoomerAMGSetSabs)} );

		parameters.insert( {"distance_R", parameter_data(2.0, & HYPRE_BoomerAMGSetRestriction)} );
		parameters.insert( {"strength_tolC", parameter_data(0.25, & HYPRE_BoomerAMGSetStrongThreshold)} );
		parameters.insert( {"strength_tolR", parameter_data(0.1, & HYPRE_BoomerAMGSetStrongThresholdR)} );
		parameters.insert( {"filterA_tol", parameter_data(1.0e-4, & HYPRE_BoomerAMGSetADropTol)} );
		parameters.insert( {"post_filter_R", parameter_data(0.0, & HYPRE_BoomerAMGSetFilterThresholdR)} );

		std::pair<std::string,std::string> relaxation_order("A","FFF");

		parameters.insert( {"relaxation_order", parameter_data( relaxation_order, &set_relaxation_order )} );

		break;
	}
	case CLASSICAL_AMG:
		parameters.insert( {"coarsen_type", parameter_data(6, & HYPRE_BoomerAMGSetCoarsenType)} );
		parameters.insert( {"relax_type", parameter_data(6, & HYPRE_BoomerAMGSetRelaxType)} );
		parameters.insert( {"max_amg_levels", parameter_data(40, & HYPRE_BoomerAMGSetMaxLevels)} );

		break;
	case NONE:
		break;
	}
}

void SolverBoomerAMG::solve(LA::SparseMatrix & system_matrix,LA::Vector & right_hand_side,LA::Vector &solution){

	Epetra_CrsMatrix * sys_matrix_pt=const_cast<Epetra_CrsMatrix *>(&system_matrix.trilinos_matrix());
	Ifpack_Hypre hypre_interface( sys_matrix_pt );

	Teuchos :: ParameterList parameter_list;
	parameter_list.set("Solver",Hypre_Solver::BoomerAMG);
	parameter_list.set("SolverOrPrecondition",Hypre_Chooser::Solver);
	parameter_list.set("SetPreconditioner",false);

	hypre_interface.SetParameters(parameter_list);
	SolverParameters.set_parameters(hypre_interface);

	hypre_interface.Compute()  ;

	Epetra_FEVector & ref_soln = solution.trilinos_vector();

	hypre_interface.ApplyInverse(right_hand_side.trilinos_vector(),ref_soln);

}


void BoomerAMG_PreconditionedSolver::solve(LA::SparseMatrix & system_matrix,LA::Vector & right_hand_side,LA::Vector &solution){

	Epetra_CrsMatrix * sys_matrix_pt=const_cast<Epetra_CrsMatrix *>(&system_matrix.trilinos_matrix());
	Ifpack_Hypre hypre_interface( sys_matrix_pt );

	Teuchos :: ParameterList parameter_list;
	parameter_list.set("Preconditioner",Hypre_Solver::BoomerAMG);
	parameter_list.set("Solver",solver_parameters.solver_selection);
	parameter_list.set("SolverOrPrecondition",Hypre_Chooser::Solver);
	parameter_list.set("SetPreconditioner",true);

	hypre_interface.SetParameters(parameter_list);
	BoomerAMG_precond_parameters.set_parameters(hypre_interface);
	solver_parameters.set_parameters(hypre_interface);


	hypre_interface.Initialize();

	hypre_interface.Compute()  ;

	Epetra_FEVector & ref_soln = solution.trilinos_vector();

	hypre_interface.ApplyInverse(right_hand_side.trilinos_vector(),ref_soln);

}

void ifpack_solver::solve(LA::SparseMatrix & system_matrix,LA::Vector & right_hand_side,LA::Vector &solution){

	Epetra_CrsMatrix * sys_matrix_pt=const_cast<Epetra_CrsMatrix *>(&system_matrix.trilinos_matrix());
	Ifpack_Hypre hypre_interface( sys_matrix_pt );

	Teuchos :: ParameterList parameter_list;
	parameter_list.set("Solver",solver_parameters.solver_selection);
	parameter_list.set("SolverOrPrecondition",Hypre_Chooser::Solver);
	parameter_list.set("SetPreconditioner",false);

	hypre_interface.SetParameters(parameter_list);
	solver_parameters.set_parameters(hypre_interface);

	hypre_interface.Initialize();

	hypre_interface.Compute()  ;

	Epetra_FEVector & ref_soln = solution.trilinos_vector();

	hypre_interface.ApplyInverse(right_hand_side.trilinos_vector(),ref_soln);

}

}
DEAL_II_NAMESPACE_CLOSE
