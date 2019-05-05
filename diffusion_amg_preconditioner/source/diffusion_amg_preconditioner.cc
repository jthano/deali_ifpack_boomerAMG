#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
//
//
///////////////////////////////////////////////
//
#include "BoomerAMG_solver.h"

namespace LA =  dealii::LinearAlgebraTrilinos;

using namespace dealii;


//
// Diffusion coefficients.
//
double diff_reg1=100.0 , diff_reg2=0.001;
//
// cube or box dimensions
//
double Lx=1.0,Ly=1.0,Lz=1.0;
//
// Band thickness
//
double dx=0.2;
//
//number of bands
//
int nx=3;
//
template<int dim>
double banded_diff_coef(typename DoFHandler<dim>::active_cell_iterator & cell){

	int vertices;
	if (dim==2) vertices=4;
	else vertices = 8;

	bool region2 = false;

	double r_strt,r_end;

	double L1 = (Lx-(dx*nx))/((double)(nx-1));

	for (unsigned int i=0;i<vertices;++i){

		dealii::Point<dim> location=cell->vertex(i);

		for (int n=1;n<=nx;++n){
			r_strt = (dx+L1)*( (double)(n-1) );
			r_end = r_strt + dx;

			if (location(0)>=r_strt && location(0)<=r_end){
				if (n%3==0 ){
					region2=true;
					break;
				}else if (n%3==1 && location(1) > Ly/7 && location(1)< Ly*0.6){
					region2=true;
					break;
				}else if (n%3==2 && (location(1) < Ly/4.0 || location(1)> Ly*0.75 )  ){
					region2=true;
					break;
				}

			}
		}

		if (region2)
			break;

	}

	double diffusion;
	if (region2)
		diffusion = diff_reg2;
	else
		diffusion = diff_reg1;

	return diffusion;

}
//
//
template <int dim>
class DiffusionSolverTest
{
public:
  enum solver_options {CG,JPCG,ICPCG,MLPCG,PCG,Classic_AMG,AIR_AMG};
  enum diffusion_coef_typ {CONST_DIFF, VARRYING_DIFF};
  DiffusionSolverTest (diffusion_coef_typ diff_coeff_selection);
  ~DiffusionSolverTest ();
  void run ();
private:
  void setup_system ();
  void assemble_system ();
  void solve (solver_options solver_selection);
  void refine_grid ();
  void output_results (const unsigned int cycle) const;
  MPI_Comm                                  mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;
  DoFHandler<dim>                           dof_handler;
  FE_Q<dim>                                 fe;
  IndexSet                                  locally_owned_dofs;
  IndexSet                                  locally_relevant_dofs;
  ConstraintMatrix                          constraints;
  LA::MPI::SparseMatrix                     system_matrix;
  LA::MPI::Vector                           locally_relevant_solution;
  LA::MPI::Vector                           system_rhs;
  ConditionalOStream                        pcout;
  TimerOutput                               computing_timer;
  diffusion_coef_typ diff_coeff_selection;
};
template <int dim>
DiffusionSolverTest<dim>::DiffusionSolverTest (diffusion_coef_typ diff_coeff_selection)
  :
  mpi_communicator (MPI_COMM_WORLD),
  triangulation (mpi_communicator,
                 typename Triangulation<dim>::MeshSmoothing
                 (Triangulation<dim>::smoothing_on_refinement |
                  Triangulation<dim>::smoothing_on_coarsening)),
  dof_handler (triangulation),
  fe (1),
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process(mpi_communicator)
          == 0)),
  computing_timer (mpi_communicator,
                   pcout,
                   TimerOutput::summary,
                   TimerOutput::wall_times),
  diff_coeff_selection(diff_coeff_selection)
{}
template <int dim>
DiffusionSolverTest<dim>::~DiffusionSolverTest ()
{
  dof_handler.clear ();
}
template <int dim>
void DiffusionSolverTest<dim>::setup_system ()
{
  TimerOutput::Scope t(computing_timer, "setup");
  dof_handler.distribute_dofs (fe);
  locally_owned_dofs = dof_handler.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dof_handler,
                                           locally_relevant_dofs);
  locally_relevant_solution.reinit (locally_owned_dofs,
                                    locally_relevant_dofs, mpi_communicator);
  system_rhs.reinit (locally_owned_dofs, mpi_communicator);
  constraints.clear ();
  constraints.reinit (locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints (dof_handler, constraints);
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            Functions::ZeroFunction<dim>(),
                                            constraints);
  constraints.close ();
  DynamicSparsityPattern dsp (locally_relevant_dofs);
  DoFTools::make_sparsity_pattern (dof_handler, dsp,
                                   constraints, false);
  SparsityTools::distribute_sparsity_pattern (dsp,
                                              dof_handler.n_locally_owned_dofs_per_processor(),
                                              mpi_communicator,
                                              locally_relevant_dofs);
  system_matrix.reinit (locally_owned_dofs,
                        locally_owned_dofs,
                        dsp,
                        mpi_communicator);
}
template <int dim>
void DiffusionSolverTest<dim>::assemble_system ()
{
  TimerOutput::Scope t(computing_timer, "assembly");
  const QGauss<dim>  quadrature_formula(3);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points |
                           update_JxW_values);
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
      {

    	double D = 0.0;

    	if (diff_coeff_selection == CONST_DIFF){
    		D=1.0;
    	} else{
    		D = banded_diff_coef<dim>(cell);
    	}

        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit (cell);
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          {
            const double
            rhs_value = 1.0;
			/*
              = (fe_values.quadrature_point(q_point)[1]
                 >
                 0.5+0.25*std::sin(4.0 * numbers::PI *
                                   fe_values.quadrature_point(q_point)[0])
                 ? 1 : -1); */
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                  cell_matrix(i,j) += D*(fe_values.shape_grad(i,q_point) *
                                       fe_values.shape_grad(j,q_point) *
                                       fe_values.JxW(q_point));
                cell_rhs(i) += (rhs_value *
                                fe_values.shape_value(i,q_point) *
                                fe_values.JxW(q_point));
              }
          }
        cell->get_dof_indices (local_dof_indices);
        constraints.distribute_local_to_global (cell_matrix,
                                                cell_rhs,
                                                local_dof_indices,
                                                system_matrix,
                                                system_rhs);
      }
  system_matrix.compress (VectorOperation::add);
  system_rhs.compress (VectorOperation::add);
}

template <int dim>
void DiffusionSolverTest<dim>::solve (solver_options solver_selection)
{

  TimerOutput::Scope t(computing_timer, "solve");
  LA::MPI::Vector
  completely_distributed_solution (locally_owned_dofs, mpi_communicator);

  //TrilinosWrappers::BoomerAMG_Parameters
  //AMG_parameters(TrilinosWrappers::BoomerAMG_Parameters::CLASSICAL_AMG);

  //TrilinosWrappers::SolverBoomerAMG AMG_solver(AMG_parameters);

  //CG,PCG,Classic_AMG,AIR_AMG

  //ICPCG

	switch(solver_selection)
	{
	case PCG:
	{

		TrilinosWrappers::BoomerAMGParameters AMG_parameters(TrilinosWrappers::BoomerAMGParameters::CLASSICAL_AMG, Hypre_Chooser::Preconditioner);
		TrilinosWrappers::ifpackSolverParameters Solver_params(Hypre_Solver::PCG);

		typedef TrilinosWrappers::ifpackSolverParameters::parameter_data parameter_data;
		/**
		 * deomonstrating how to add parameters
		 */
		Solver_params.add_parameter("pcg_convergence_tol", parameter_data(1.e-10,&HYPRE_ParCSRPCGSetTol) );
		Solver_params.add_parameter("pcg_max_itter", parameter_data(3000,&HYPRE_ParCSRPCGSetMaxIter) );
		Solver_params.add_parameter("pcg_print_level", parameter_data(3,&HYPRE_ParCSRPCGSetPrintLevel) );

		TrilinosWrappers::BoomerAMG_PreconditionedSolver AMG_solver(AMG_parameters,Solver_params);

		AMG_solver.solve(system_matrix, system_rhs, completely_distributed_solution);
		break;
	}
	case CG:
	{

		dealii::SolverControl solver_control(3000,1e-10);
		TrilinosWrappers::SolverCG::AdditionalData additional_data(true);
		TrilinosWrappers::SolverCG solver(solver_control, additional_data);

		solver.solve(system_matrix,completely_distributed_solution,system_rhs,TrilinosWrappers::PreconditionIdentity() );

		break;
	}
	case ICPCG:
	{

		dealii::SolverControl solver_control(3000,1e-10);
		TrilinosWrappers::SolverCG::AdditionalData additional_data(true);
		TrilinosWrappers::SolverCG solver(solver_control, additional_data);

		TrilinosWrappers::PreconditionIC ic_prcond;

		ic_prcond.initialize(system_matrix);

		solver.solve(system_matrix,completely_distributed_solution,system_rhs, ic_prcond);

		break;
	}
	case Classic_AMG:
	{
	    TrilinosWrappers::BoomerAMGParameters AMG_parameters(TrilinosWrappers::BoomerAMGParameters::CLASSICAL_AMG, Hypre_Chooser::Solver);

		TrilinosWrappers::SolverBoomerAMG AMG_solver(AMG_parameters);

		AMG_solver.solve(system_matrix, system_rhs, completely_distributed_solution);

		break;
	}

	case MLPCG:
	{
		SolverControl solver_control (3000, 1e-12);
		TrilinosWrappers::SolverCG::AdditionalData additional_data(true);
		LA::SolverCG solver(solver_control,additional_data);

	    LA::MPI::PreconditionAMG preconditioner;
	    LA::MPI::PreconditionAMG::AdditionalData data;

	    preconditioner.initialize(system_matrix);
    	solver.solve (system_matrix, completely_distributed_solution, system_rhs,
    	              preconditioner);
	}

	}
	constraints.distribute (completely_distributed_solution);
	locally_relevant_solution = completely_distributed_solution;

}

template <int dim>
void DiffusionSolverTest<dim>::refine_grid ()
{
  TimerOutput::Scope t(computing_timer, "refine");

	dealii::Point<dim> lower_left_pt, upper_right_pt;

	if (dim == 2){
		lower_left_pt[0] = 0.0;
		lower_left_pt[1] = 0.0;
		upper_right_pt[0] = 1.0;
		upper_right_pt[1] = 1.0;
	} else{
		lower_left_pt[0] = 0.0;
		lower_left_pt[1] = 0.0;
		lower_left_pt[2] = 0.0;
		upper_right_pt[0] = 1.0;
		upper_right_pt[1] = 1.0;
		upper_right_pt[2] = 1.0;
	}

	std::vector<unsigned int> repetitions(dim);

	for (int i=0;i<repetitions.size();++i)
		repetitions[i] = 2;

	dealii::GridGenerator::subdivided_hyper_rectangle<dim,dim>(triangulation , repetitions , lower_left_pt , upper_right_pt);

  Vector<float> dummy_error;

  triangulation.refine_global(1);
  //8 gets about 1e6 cells, 6 solvers all competitive
  for(int i=0;i<7;++i){
  	dummy_error.reinit(triangulation.n_active_cells());
  	dummy_error = 1.0;
  	parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation,dummy_error,1.0,0.0);
  	triangulation.execute_coarsening_and_refinement();
  }

  pcout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;

}
template <int dim>
void DiffusionSolverTest<dim>::output_results (const unsigned int cycle) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (locally_relevant_solution, "u");
  Vector<float> subdomain (triangulation.n_active_cells());
  for (unsigned int i=0; i<subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector (subdomain, "subdomain");
  data_out.build_patches ();
  const std::string filename = ("solution-" +
                                Utilities::int_to_string (cycle, 2) +
                                "." +
                                Utilities::int_to_string
                                (triangulation.locally_owned_subdomain(), 4));
  std::ofstream output ((filename + ".vtu").c_str());
  data_out.write_vtu (output);
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i=0;
           i<Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++i)
        filenames.push_back ("solution-" +
                             Utilities::int_to_string (cycle, 2) +
                             "." +
                             Utilities::int_to_string (i, 4) +
                             ".vtu");
      std::ofstream master_output (("solution-" +
                                    Utilities::int_to_string (cycle, 2) +
                                    ".pvtu").c_str());
      data_out.write_pvtu_record (master_output, filenames);
    }
}
template <int dim>
void DiffusionSolverTest<dim>::run ()
{

    pcout << "Running with Trilinos on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

  refine_grid ();

  pcout << "Solver CG"<< std::endl;
  setup_system ();
  assemble_system ();
  solve (DiffusionSolverTest<2>::CG);
  computing_timer.print_summary ();
  computing_timer.reset ();
  pcout << std::endl;

  pcout << "Solver ICPCG"<< std::endl;
  setup_system ();
  assemble_system ();
  solve (DiffusionSolverTest<2>::ICPCG);
  computing_timer.print_summary ();
  computing_timer.reset ();
  pcout << std::endl;

  pcout << "Solver PCG"<< std::endl;
  setup_system ();
  assemble_system ();
  solve (DiffusionSolverTest<2>::PCG);
  computing_timer.print_summary ();
  computing_timer.reset ();
  pcout << std::endl;

  pcout << "BoomerAMG solver"<< std::endl;
  setup_system ();
  assemble_system ();
  solve (DiffusionSolverTest<2>::Classic_AMG);
  computing_timer.print_summary ();
  computing_timer.reset ();
  pcout << std::endl;

  pcout << "MLPCG"<< std::endl;
  setup_system ();
  assemble_system ();
  solve (DiffusionSolverTest<2>::MLPCG);
  computing_timer.print_summary ();
  computing_timer.reset ();
  pcout << std::endl;

  output_results (1);
}


int main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      DiffusionSolverTest<2> laplace_problem_2d(DiffusionSolverTest<2>::VARRYING_DIFF);
      laplace_problem_2d.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
