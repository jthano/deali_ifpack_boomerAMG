/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2018 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/geometry_info.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/grid/grid_tools.h>

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

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
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include "BoomerAMG_solver.h"

#include <fstream>
#include <iostream>

#include <math.h>

using namespace dealii;


class Advection_Diffusion
{
public:

  enum boundary_condition_type {HOMOGENEOUS_DIRICHLET, HOMGENEOUS_NATURAL};
  enum solver_option {DIRECT, AIR_AMG, CLASSIC_AMG};

  Advection_Diffusion(boundary_condition_type bc_type, solver_option solver_type, bool stabilize);

  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void assemble_system_stabilized();
  void solve();
  void output_results() const;

  MPI_Comm mpi_communicator;
  parallel::distributed::Triangulation<2> triangulation;

  FE_Q<2>          fe;
  DoFHandler<2>    dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  LA::MPI::SparseMatrix system_matrix;
  LA::MPI::Vector       locally_relevant_solution;
  LA::MPI::Vector       system_rhs;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;

  Point<2> velocity;
  const double speed = 100;
  const double nu;

  boundary_condition_type bc_type;
  solver_option solver_type;
  bool stabilize;

};


Advection_Diffusion::Advection_Diffusion(boundary_condition_type bc_type, solver_option solver_type, bool stabilize)
: mpi_communicator(MPI_COMM_WORLD)
, triangulation(mpi_communicator,
                typename Triangulation<2>::MeshSmoothing(
                  Triangulation<2>::smoothing_on_refinement |
                  Triangulation<2>::smoothing_on_coarsening))
, dof_handler(triangulation)
, fe(1)
, pcout(std::cout,
        (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
, computing_timer(mpi_communicator,
                  pcout,
                  TimerOutput::summary,
                  TimerOutput::wall_times)
,	nu(1.0)
, bc_type(bc_type)
, solver_type(solver_type)
, stabilize(stabilize)
{
	  velocity(0) = pow(2.0,0.5)/2.0;//0.15;//pow(2.0,0.5)/2.0;
	  velocity(1) = pow(2.0,0.5)/2.0;//0.9886859966642595;//velocity(0);

	  velocity(0) *= speed;
	  velocity(1) *= speed;
}



void Advection_Diffusion::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(1);

  Vector<float> dummy_error;

  //9
  for(int i=0;i<4;++i){
  	dummy_error.reinit(triangulation.n_active_cells());
  	dummy_error = 1.0;
  	parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation,dummy_error,1.0,0.0);
  	triangulation.execute_coarsening_and_refinement();
  }

  //GridTools::distort_random(0.2, triangulation);

  pcout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}




void Advection_Diffusion::setup_system()
{



    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);


    /**
     * Here the ability to set natural boundary conditions are added. That is, the inlfow
     * boundary condition is treated as Dirichlet and the outflow boundary is treated by specifying
     * a diffusive flux. For simplicity, both values are currently hardcoded as 0.0
     */
    if (bc_type == HOMGENEOUS_NATURAL)
    {
		QGauss<1> face_quadrature_formula(1);

		FEFaceValues<2> fe_face_values (fe, face_quadrature_formula,
										  update_quadrature_points  |
										  update_normal_vectors );

		/**
		 * Loop through all relevant cells and find outflow boundaries and mark these
		 * with a boundary_id other than 0. This will prevent interpolate_boundary_values
		 * called after this loop from applying Dirichlet boundary conditions at these
		 * faces.
		 */

		for (const auto &cell : dof_handler.active_cell_iterators())
		  {
			if (cell->is_locally_owned() || cell->is_ghost())
			  {

				for (unsigned int face_number = 0;
					 face_number < GeometryInfo<2>::faces_per_cell;
					 ++face_number)
				  if (cell->face(face_number)->at_boundary() )
					{
					  fe_face_values.reinit(cell, face_number);

					  double beta = velocity*fe_face_values.normal_vector(0);

					  if (beta>=0.0){
						  cell->face(face_number)->set_boundary_id(1);
					  }

					}

			  }
		  }

    }

    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<2>(),
                                             constraints);
    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(
      dsp,
      dof_handler.n_locally_owned_dofs_per_processor(),
      mpi_communicator,
      locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
}

/**
 * This function is essentially the same as that from step 40 except that an advection
 * term has been added. Additionally, linear elements are being used.
 */
void Advection_Diffusion::assemble_system(){

	QGauss<2> quadrature_formula(2);

	FEValues<2> fe_values(fe,
	                      quadrature_formula,
	                      update_values | update_gradients | update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points    = quadrature_formula.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double>     cell_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	/**
	 * The assembly processes is the same from step 40 except and advection term
	 * is added and the option for a variable diffusion coefficient nu is added
	 */
	  for (const auto &cell : dof_handler.active_cell_iterators())
	    {
	      if (cell->is_locally_owned())
	        {

			  cell_matrix = 0;
			  cell_rhs    = 0;

			  fe_values.reinit(cell);

			  for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
				{
				  for (unsigned int i = 0; i < dofs_per_cell; ++i)
					for (unsigned int j = 0; j < dofs_per_cell; ++j){
					  cell_matrix(i, j) +=
						nu*(fe_values.shape_grad(i, q_index) *
						 fe_values.shape_grad(j, q_index) *
						 fe_values.JxW(q_index));

					  cell_matrix(i, j) +=
						(velocity * fe_values.shape_grad(j, q_index)) *
						 fe_values.shape_value(i, q_index) *
						 fe_values.JxW(q_index);

					}
				  for (unsigned int i = 0; i < dofs_per_cell; ++i)
				  {
					cell_rhs(i) += (fe_values.shape_value(i, q_index) *
									1.0 * fe_values.JxW(q_index));
				  }
				}

			  cell->get_dof_indices(local_dof_indices);

	          cell->get_dof_indices(local_dof_indices);
	          constraints.distribute_local_to_global(cell_matrix,
	                                                 cell_rhs,
	                                                 local_dof_indices,
	                                                 system_matrix,
	                                                 system_rhs);
	        }
	    }

	  system_matrix.compress(VectorOperation::add);
	  system_rhs.compress(VectorOperation::add);


}
/**
 * This function implements an SUPG discretization. The specifics of the stabilization
 * are presented in the report.
 */
void Advection_Diffusion::assemble_system_stabilized()
{
  bool stabilize = true;

  QGauss<2> quadrature_formula(2);

  FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);


  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
		  fe_values.reinit(cell);
		  //
		  Point<2> zero = fe_values.get_mapping().transform_unit_to_real_cell(cell, Point<2>(0.0,0.0));
		  Point<2> n_xi = fe_values.get_mapping().transform_unit_to_real_cell(cell, Point<2>(1.0,0.0));
		  n_xi -= zero;
		  n_xi /= n_xi.norm();
		  //
		  Point<2> n_etta = fe_values.get_mapping().transform_unit_to_real_cell(cell, Point<2>(0.0,1.0));
		  n_etta -= zero;
		  n_etta /= n_etta.norm();
		  //
		  cell_matrix = 0;
		  cell_rhs    = 0;
		  //
		  // Compute P(w)tau
		  //
		  Point<2> x0,x1,x2,x3;
		  //
		  x0 = cell->vertex(0);
		  x1 = cell->vertex(1);
		  x2 = cell->vertex(2);
		  x3 = cell->vertex(3);
		  //
		  double h_xi = ( x3(0) + x1(0) - x2(0) - x0(0) )/2.0;
		  double h_etta = ( x3(1) + x2(1) - x0(1) - x1(1) )/2.0;
		  //
		  double a_xi = velocity*n_xi;
		  double a_etta = velocity*n_etta;
		  //
		  double Pe_xi = a_xi*h_xi/2.0/nu;
		  double Pe_etta = a_etta*h_etta/2.0/nu;
		  //
		  double etta_bar = 1.0/tanh(Pe_xi) - 1.0/Pe_xi;
		  double xi_bar = 1.0/tanh(Pe_etta) - 1.0/Pe_etta;
		  //
		  double tau = (etta_bar*a_etta*h_etta + xi_bar*a_xi*h_xi)/2.0/pow(velocity.norm(),2);
		  //
		  for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
			{
			  for (unsigned int i = 0; i < dofs_per_cell; ++i)
				for (unsigned int j = 0; j < dofs_per_cell; ++j){
				  cell_matrix(i, j) +=
					nu*(fe_values.shape_grad(i, q_index) *
					 fe_values.shape_grad(j, q_index) *
					 fe_values.JxW(q_index));

				  cell_matrix(i, j) +=
					(velocity * fe_values.shape_grad(j, q_index) *
					 fe_values.shape_value(i, q_index) *
					 fe_values.JxW(q_index));

				  cell_matrix(i,j) += (fe_values.shape_grad(i,q_index)*
						  velocity)*tau*(fe_values.shape_grad(j,q_index)*velocity)
						  *fe_values.JxW(q_index);

				}
			  for (unsigned int i = 0; i < dofs_per_cell; ++i)
			  {
				cell_rhs(i) += (fe_values.shape_value(i, q_index) *
								1.0 * fe_values.JxW(q_index));
				cell_rhs(i) += (fe_values.shape_grad(i,q_index)*velocity)*tau*fe_values.JxW(q_index);
			  }
			}


		  cell->get_dof_indices(local_dof_indices);

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

}



void Advection_Diffusion::solve()
{
    TimerOutput::Scope t(computing_timer, "solve");

    LA::MPI::Vector    completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    if (solver_type == AIR_AMG){
    	TrilinosWrappers::BoomerAMGParameters AMG_parameters(TrilinosWrappers::BoomerAMGParameters::AIR_AMG, Hypre_Chooser::Solver);
        /**
         * Demonstrate changing a parameter value
         */
    	AMG_parameters.set_parameter_value("relax_type",3);
    	TrilinosWrappers::SolverBoomerAMG AMG_solver(AMG_parameters);
    	AMG_solver.solve(system_matrix, system_rhs, completely_distributed_solution);
    	AMG_solver.solve(system_matrix, system_rhs, completely_distributed_solution);

    }else if (solver_type == CLASSIC_AMG){
    	TrilinosWrappers::BoomerAMGParameters AMG_parameters(TrilinosWrappers::BoomerAMGParameters::CLASSICAL_AMG, Hypre_Chooser::Solver);
    	TrilinosWrappers::SolverBoomerAMG AMG_solver(AMG_parameters);
    	AMG_solver.solve(system_matrix, system_rhs, completely_distributed_solution);
    } else{
        SolverControl solver_control(3000,1e-6);
    	TrilinosWrappers::SolverDirect Solv(solver_control);
    	Solv.initialize(system_matrix);
    	Solv.solve(completely_distributed_solution,system_rhs);

    }

    constraints.distribute(completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;

}



void Advection_Diffusion::output_results() const
{
	int cycle=1;

    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "u");

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    const std::string filename =
      ("solution-" + Utilities::int_to_string(cycle, 2) + "." +
       Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
    std::ofstream output(filename + ".vtu");
    data_out.write_vtu(output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i = 0;
             i < Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          filenames.push_back("solution-" + Utilities::int_to_string(cycle, 2) +
                              "." + Utilities::int_to_string(i, 4) + ".vtu");

        std::ofstream master_output(
          "solution-" + Utilities::int_to_string(cycle, 2) + ".pvtu");
        data_out.write_pvtu_record(master_output, filenames);
      }

}



void Advection_Diffusion::run()
{
  make_grid();
  setup_system();
  assemble_system_stabilized();
  //assemble_system();
  solve();
  output_results();
}



int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  deallog.depth_console(2);

  Advection_Diffusion laplace_problem(Advection_Diffusion::HOMOGENEOUS_DIRICHLET,Advection_Diffusion::DIRECT, true );
  laplace_problem.run();

  return 0;
}
