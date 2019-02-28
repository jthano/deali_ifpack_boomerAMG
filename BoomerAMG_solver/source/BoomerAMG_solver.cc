
#include<BoomerAMG_solver.h>

DEAL_II_NAMESPACE_OPEN

namespace TrilinosWrappers
{

void BoomerAMG_Parameters::pre_post_relax::set_parameter(Hypre_Chooser solver_preconditioner_selection,std::vector<FunctionParameter> * parameter_list){

	if (parameter_used)
	{
		const unsigned int ns_down = value.first.length();
		const unsigned int ns_up = value.second.length();
		const unsigned int ns_coarse = 1 ;

		const std::string F("F");
		const std::string C("C");
		const std::string A("A");

		// Array to store relaxation scheme and pass to Hypre
		int **grid_relax_points = (int **) malloc(4*sizeof(int *));
		grid_relax_points[0] = NULL;
		grid_relax_points[1] = (int *) malloc(sizeof(int)*ns_down);
		grid_relax_points[2] = (int *) malloc(sizeof(int)*ns_up);
		grid_relax_points[3] = (int *) malloc(sizeof(int));
		grid_relax_points[3][0] = 0;

		// set down relax scheme
		for(unsigned int i = 0; i<ns_down; i++) {
		    if (value.first.compare(i,1,F) == 0) {
		       grid_relax_points[1][i] = -1;
		    }
		    else if (value.first.compare(i,1,C) == 0) {
		       grid_relax_points[1][i] = 1;
		    }
		    else if (value.first.compare(i,1,A) == 0) {
		       grid_relax_points[1][i] = 0;
		    }
		 }

		 // set up relax scheme
		 for(unsigned int i = 0; i<ns_up; i++) {
		    if (value.second.compare(i,1,F) == 0) {
		       grid_relax_points[2][i] = -1;
		    }
		    else if (value.second.compare(i,1,C) == 0) {
		       grid_relax_points[2][i] = 1;
		    }
		    else if (value.second.compare(i,1,A) == 0) {
		       grid_relax_points[2][i] = 0;
		    }
		 }
		 // Currently some problem with Trilinos I thought I fixed. Here hypre needs an interger pointer to list of intergers
		 /*
		FunctionParameter f1 (solver_preconditioner_selection , & HYPRE_BoomerAMGSetGridRelaxPoints , ns_coarse,3);
		FunctionParameter f2 (solver_preconditioner_selection , & HYPRE_BoomerAMGSetCycleNumSweeps , ns_coarse,3);
		FunctionParameter f3 (solver_preconditioner_selection , & HYPRE_BoomerAMGSetCycleNumSweeps , ns_down,1);
		FunctionParameter f4 (solver_preconditioner_selection , & HYPRE_BoomerAMGSetCycleNumSweeps , ns_up,2);

		parameter_list->push_back(f1);
		parameter_list->push_back(f2);
		parameter_list->push_back(f3);
		parameter_list->push_back(f4); */
	}

}

void BoomerAMG_Parameters::relax_type::set_parameter(Hypre_Chooser solver_preconditioner_selection, std::vector<FunctionParameter> * parameter_list){

	if (parameter_used)
	{
		FunctionParameter function(solver_preconditioner_selection , & HYPRE_BoomerAMGSetRelaxType , value);
		parameter_list->push_back(function);
	}

}

void BoomerAMG_Parameters::interp_type::set_parameter(Hypre_Chooser solver_preconditioner_selection, std::vector<FunctionParameter> * parameter_list){

	if (parameter_used)
	{
		FunctionParameter function(solver_preconditioner_selection , & HYPRE_BoomerAMGSetInterpType , value);
		parameter_list->push_back(function);
	}

}

void BoomerAMG_Parameters::coarsen_type::set_parameter(Hypre_Chooser solver_preconditioner_selection, std::vector<FunctionParameter> * parameter_list){

	if (parameter_used)
	{
		FunctionParameter function(solver_preconditioner_selection , & HYPRE_BoomerAMGSetCoarsenType , value);
		parameter_list->push_back(function);
	}

}

void BoomerAMG_Parameters::print_level::set_parameter(Hypre_Chooser solver_preconditioner_selection, std::vector<FunctionParameter> * parameter_list){

	if (parameter_used)
	{
		FunctionParameter function(solver_preconditioner_selection , & HYPRE_BoomerAMGSetPrintLevel , value);
		parameter_list->push_back(function);
	}

}

void BoomerAMG_Parameters::max_levels::set_parameter(Hypre_Chooser solver_preconditioner_selection, std::vector<FunctionParameter> * parameter_list){

	if (parameter_used)
	{
		FunctionParameter function(solver_preconditioner_selection , & HYPRE_BoomerAMGSetMaxLevels , value);
		parameter_list->push_back(function);
	}

}

void BoomerAMG_Parameters::cycle_type::set_parameter(Hypre_Chooser solver_preconditioner_selection, std::vector<FunctionParameter> * parameter_list){

	if (parameter_used)
	{
		FunctionParameter function(solver_preconditioner_selection , & HYPRE_BoomerAMGSetCycleType , value);
		parameter_list->push_back(function);
	}

}

void BoomerAMG_Parameters::sabs_flag::set_parameter(Hypre_Chooser solver_preconditioner_selection, std::vector<FunctionParameter> * parameter_list){

	if (parameter_used)
	{
		FunctionParameter function(solver_preconditioner_selection , & HYPRE_BoomerAMGSetSabs , value);
		parameter_list->push_back(function);
	}

}

void BoomerAMG_Parameters::strength_tolC::set_parameter(Hypre_Chooser solver_preconditioner_selection, std::vector<FunctionParameter> * parameter_list){

	if (parameter_used)
	{
		FunctionParameter function(solver_preconditioner_selection , & HYPRE_BoomerAMGSetStrongThreshold , value);
		parameter_list->push_back(function);
	}

}

void BoomerAMG_Parameters::strength_tolR::set_parameter(Hypre_Chooser solver_preconditioner_selection, std::vector<FunctionParameter> * parameter_list){

	if (parameter_used)
	{
		FunctionParameter function(solver_preconditioner_selection , & HYPRE_BoomerAMGSetStrongThresholdR , value);
		parameter_list->push_back(function);
	}

}

void BoomerAMG_Parameters::distance_R::set_parameter(Hypre_Chooser solver_preconditioner_selection, std::vector<FunctionParameter> * parameter_list){

	if (parameter_used)
	{
		FunctionParameter function(solver_preconditioner_selection , & HYPRE_BoomerAMGSetRestriction , value);
		parameter_list->push_back(function);
	}

}

void BoomerAMG_Parameters::filterA_tol::set_parameter(Hypre_Chooser solver_preconditioner_selection, std::vector<FunctionParameter> * parameter_list){

	if (parameter_used)
	{
		FunctionParameter function(solver_preconditioner_selection , & HYPRE_BoomerAMGSetADropTol , value);
		parameter_list->push_back(function);
	}

}

void BoomerAMG_Parameters::set_parameter_list(std::vector<FunctionParameter> * parameter_vector){
	for (auto parameter : BoomerAMG_Parameter_Base::parameter_list){
		parameter->set_parameter( solver_preconditioner_selection, parameter_vector );
	}
}


} // Close namespace TrilinosWrappers
DEAL_II_NAMESPACE_CLOSE
