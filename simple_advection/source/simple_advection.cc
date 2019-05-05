/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2018 by the deal.II authors
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
 * Author: Guido Kanschat, Texas A&M University, 2009
 */
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/numerics/derivative_approximation.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <iostream>
#include <fstream>
//
//
///////////////////////////////////////////////
//
#include "BoomerAMG_solver.h"


namespace LA =  dealii::LinearAlgebraTrilinos;

using namespace dealii;

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues() = default;
  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<double> &          values,
                          const unsigned int component = 0) const override;
};

template <int dim>
void BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points,
                                     std::vector<double> &          values,
                                     const unsigned int component) const
{
  (void)component;
  AssertIndexRange(component, 1);
  Assert(values.size() == points.size(),
         ExcDimensionMismatch(values.size(), points.size()));

  for (unsigned int i = 0; i < values.size(); ++i)
    {
      if (points[i](0) < 0.5)
        values[i] = 1.;
      else
        values[i] = 0.;
    }
}


template <int dim>
Tensor<1, dim> beta(const Point<dim> &p)
{
  Assert(dim >= 2, ExcNotImplemented());

  Point<dim> wind_field;
  wind_field(0) = -p(1);
  wind_field(1) = p(0);
  wind_field /= wind_field.norm();

  return wind_field;
}



template <int dim>
class AdvectionProblem
{
public:
  AdvectionProblem();
  void run();

private:
  void precondition(LA::MPI::SparseMatrix & system_matrix, LA::MPI::Vector & right_hand_side);
  void setup_system();
  void assemble_system();
  void solve(LA::MPI::Vector &solution);
  void refine_grid();
  void output_results(const unsigned int cycle) const;

  MPI_Comm                                  mpi_communicator;

  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  ConditionalOStream                pcout;

  parallel::distributed::Triangulation<dim>   triangulation;
  const MappingQ1<dim> mapping;

  FE_DGQ<dim>     fe;
  DoFHandler<dim> dof_handler;

  LA::MPI::SparseMatrix system_matrix;

  LA::MPI::Vector solution;
  LA::MPI::Vector right_hand_side;

  using DoFInfo  = MeshWorker::DoFInfo<dim>;
  using CellInfo = MeshWorker::IntegrationInfo<dim>;

  static void integrate_cell_term(DoFInfo &dinfo, CellInfo &info);
  static void integrate_boundary_term(DoFInfo &dinfo, CellInfo &info);
  static void integrate_face_term(DoFInfo & dinfo1,
                                  DoFInfo & dinfo2,
                                  CellInfo &info1,
                                  CellInfo &info2);
};

template <int dim>
AdvectionProblem<dim>::AdvectionProblem()
  : mpi_communicator (MPI_COMM_WORLD),
	n_mpi_processes (dealii::Utilities::MPI::n_mpi_processes(mpi_communicator)),
	this_mpi_process (dealii::Utilities::MPI::this_mpi_process(mpi_communicator)),
	pcout (std::cout,
	       (this_mpi_process == 0)),
    triangulation(mpi_communicator,typename Triangulation<dim>::MeshSmoothing
            (Triangulation<dim>::smoothing_on_refinement |
             Triangulation<dim>::smoothing_on_coarsening)),
	mapping(),
	fe(1),
	dof_handler(triangulation)

{}

template <int dim>
void AdvectionProblem<dim>::precondition(LA::MPI::SparseMatrix & system_matrix, LA::MPI::Vector & right_hand_side)
{
	int i_block_size;
	if (dim == 2)
		i_block_size=4;
	else
		i_block_size=8;

	dealii::FullMatrix<double> diagonal_block(i_block_size,i_block_size);

	LA::MPI::SparseMatrix preconditioner;

	preconditioner.reinit(system_matrix);

	preconditioner = 0.0;

	const std::pair<int, int>  local_range = system_matrix.local_range();

	const int off_set = local_range.first;

	for (unsigned int i_block =0; i_block<system_matrix.local_size() ;i_block+=i_block_size){

		for (unsigned int i = 0; i<i_block_size ; ++i) {

			for (unsigned int j = 0; j<i_block_size ; ++j) {
				diagonal_block[i][j] = system_matrix(i+i_block+off_set , j+i_block+off_set);
			}
		}

		diagonal_block.gauss_jordan();

		for (unsigned int i = 0; i<i_block_size ; ++i) {

			for (unsigned int j = 0; j<i_block_size ; ++j) {
				preconditioner.set(i+i_block+off_set , j+i_block+off_set ,diagonal_block[i][j]);
			}
		}
	}
	//
	LA::MPI::SparseMatrix store_solution;
	//
	store_solution.reinit(system_matrix.trilinos_matrix());
	//
	preconditioner.mmult(system_matrix,store_solution);
	//
	LA::MPI::Vector rhs_store;
	rhs_store = right_hand_side;
	//
	preconditioner.vmult(right_hand_side,rhs_store);
}

template <int dim>
void AdvectionProblem<dim>::setup_system()
{
	//
	dof_handler.distribute_dofs (fe);
	//
    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    //
    //
    IndexSet locally_relevant_dofs ;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    //
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    //
    DoFTools::make_flux_sparsity_pattern (dof_handler, dsp);
    //
    SparsityTools::distribute_sparsity_pattern(
      dsp,
	  dof_handler.n_locally_owned_dofs_per_processor(),
      mpi_communicator,
      locally_relevant_dofs);
    //
    system_matrix.reinit (locally_owned_dofs,
                          locally_owned_dofs,
						  dsp,
    					  mpi_communicator);
    //
    //
    solution.reinit(locally_owned_dofs, mpi_communicator);
    right_hand_side.reinit(locally_owned_dofs, mpi_communicator);
}


template <int dim>
void AdvectionProblem<dim>::assemble_system()
{
  MeshWorker::IntegrationInfoBox<dim> info_box;

  const unsigned int n_gauss_points = dof_handler.get_fe().degree + 1;
  info_box.initialize_gauss_quadrature(n_gauss_points,
                                       n_gauss_points,
                                       n_gauss_points);

  info_box.initialize_update_flags();
  UpdateFlags update_flags =
    update_quadrature_points | update_values | update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);

  info_box.initialize(fe, mapping);

  MeshWorker::DoFInfo<dim> dof_info(dof_handler);

  MeshWorker::Assembler::SystemSimple<LA::MPI::SparseMatrix, LA::MPI::Vector>
    assembler;
  assembler.initialize(system_matrix, right_hand_side);

  MeshWorker::LoopControl loop_control;

  loop_control.faces_to_ghost = dealii::MeshWorker::LoopControl::both;

  MeshWorker::loop<dim,
                   dim,
                   MeshWorker::DoFInfo<dim>,
                   MeshWorker::IntegrationInfoBox<dim>>(
    dof_handler.begin_active(),
    dof_handler.end(),
    dof_info,
    info_box,
    &AdvectionProblem<dim>::integrate_cell_term,
    &AdvectionProblem<dim>::integrate_boundary_term,
    &AdvectionProblem<dim>::integrate_face_term,
    assembler,loop_control);
}



template <int dim>
void AdvectionProblem<dim>::integrate_cell_term(DoFInfo & dinfo,
                                                CellInfo &info)
{
  const FEValuesBase<dim> &  fe_values    = info.fe_values();
  FullMatrix<double> &       local_matrix = dinfo.matrix(0).matrix;
  const std::vector<double> &JxW          = fe_values.get_JxW_values();

  for (unsigned int point = 0; point < fe_values.n_quadrature_points; ++point)
    {
      const Tensor<1, dim> beta_at_q_point =
        beta(fe_values.quadrature_point(point));

      for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
        for (unsigned int j = 0; j < fe_values.dofs_per_cell; ++j)
          local_matrix(i, j) += -beta_at_q_point *                //
                                fe_values.shape_grad(i, point) *  //
                                fe_values.shape_value(j, point) * //
                                JxW[point];
    }
}

template <int dim>
void AdvectionProblem<dim>::integrate_boundary_term(DoFInfo & dinfo,
                                                    CellInfo &info)
{
  const FEValuesBase<dim> &fe_face_values = info.fe_values();
  FullMatrix<double> &     local_matrix   = dinfo.matrix(0).matrix;
  Vector<double> &         local_vector   = dinfo.vector(0).block(0);

  const std::vector<double> &        JxW = fe_face_values.get_JxW_values();
  const std::vector<Tensor<1, dim>> &normals =
    fe_face_values.get_normal_vectors();

  std::vector<double> g(fe_face_values.n_quadrature_points);

  static BoundaryValues<dim> boundary_function;
  boundary_function.value_list(fe_face_values.get_quadrature_points(), g);

  for (unsigned int point = 0; point < fe_face_values.n_quadrature_points;
       ++point)
    {
      const double beta_dot_n =
        beta(fe_face_values.quadrature_point(point)) * normals[point];
      if (beta_dot_n > 0)
        for (unsigned int i = 0; i < fe_face_values.dofs_per_cell; ++i)
          for (unsigned int j = 0; j < fe_face_values.dofs_per_cell; ++j)
            local_matrix(i, j) += beta_dot_n *                           //
                                  fe_face_values.shape_value(j, point) * //
                                  fe_face_values.shape_value(i, point) * //
                                  JxW[point];
      else
        for (unsigned int i = 0; i < fe_face_values.dofs_per_cell; ++i)
          local_vector(i) += -beta_dot_n *                          //
                             g[point] *                             //
                             fe_face_values.shape_value(i, point) * //
                             JxW[point];
    }
}

template <int dim>
void AdvectionProblem<dim>::integrate_face_term(DoFInfo & dinfo1,
                                                DoFInfo & dinfo2,
                                                CellInfo &info1,
                                                CellInfo &info2)
{
  const FEValuesBase<dim> &fe_face_values = info1.fe_values();
  const unsigned int       dofs_per_cell  = fe_face_values.dofs_per_cell;

  const FEValuesBase<dim> &fe_face_values_neighbor = info2.fe_values();
  const unsigned int       neighbor_dofs_per_cell =
    fe_face_values_neighbor.dofs_per_cell;

  FullMatrix<double> &u1_v1_matrix = dinfo1.matrix(0, false).matrix;
  FullMatrix<double> &u2_v1_matrix = dinfo1.matrix(0, true).matrix;
  FullMatrix<double> &u1_v2_matrix = dinfo2.matrix(0, true).matrix;
  FullMatrix<double> &u2_v2_matrix = dinfo2.matrix(0, false).matrix;


  const std::vector<double> &        JxW = fe_face_values.get_JxW_values();
  const std::vector<Tensor<1, dim>> &normals =
    fe_face_values.get_normal_vectors();

  for (unsigned int point = 0; point < fe_face_values.n_quadrature_points;
       ++point)
    {
      const double beta_dot_n =
        beta(fe_face_values.quadrature_point(point)) * normals[point];
      if (beta_dot_n > 0)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              u1_v1_matrix(i, j) += beta_dot_n *                           //
                                    fe_face_values.shape_value(j, point) * //
                                    fe_face_values.shape_value(i, point) * //
                                    JxW[point];

          for (unsigned int k = 0; k < neighbor_dofs_per_cell; ++k)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              u1_v2_matrix(k, j) +=
                -beta_dot_n *                                   //
                fe_face_values.shape_value(j, point) *          //
                fe_face_values_neighbor.shape_value(k, point) * //
                JxW[point];
        }
      else
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int l = 0; l < neighbor_dofs_per_cell; ++l)
              u2_v1_matrix(i, l) +=
                beta_dot_n *                                    //
                fe_face_values_neighbor.shape_value(l, point) * //
                fe_face_values.shape_value(i, point) *          //
                JxW[point];

          for (unsigned int k = 0; k < neighbor_dofs_per_cell; ++k)
            for (unsigned int l = 0; l < neighbor_dofs_per_cell; ++l)
              u2_v2_matrix(k, l) +=
                -beta_dot_n *                                   //
                fe_face_values_neighbor.shape_value(l, point) * //
                fe_face_values_neighbor.shape_value(k, point) * //
                JxW[point];
        }
    }
}

template <int dim>
void AdvectionProblem<dim>::solve(LA::MPI::Vector &solution)
{

	precondition(system_matrix, right_hand_side);

	TrilinosWrappers::BoomerAMGParameters AMG_parameters(TrilinosWrappers::BoomerAMGParameters::AIR_AMG, Hypre_Chooser::Solver);
    /**
     * Demonstrate changing a parameter value
     */
	AMG_parameters.set_parameter_value("distance_R",1.0);
	TrilinosWrappers::SolverBoomerAMG AMG_solver(AMG_parameters);

	AMG_solver.solve(system_matrix, right_hand_side, solution);

}


template <int dim>
void AdvectionProblem<dim>::refine_grid()
{
  Vector<float> gradient_indicator(triangulation.n_active_cells());

  DerivativeApproximation::approximate_gradient(mapping,
                                                dof_handler,
                                                solution,
                                                gradient_indicator);

  unsigned int cell_no = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    gradient_indicator(cell_no++) *=
      std::pow(cell->diameter(), 1 + 1.0 * dim / 2);

  parallel::distributed::GridRefinement::
  refine_and_coarsen_fixed_number (triangulation,
		  gradient_indicator,
                                   0.3, 0.03);
  triangulation.execute_coarsening_and_refinement ();

}


template <int dim>
void AdvectionProblem<dim>::output_results (const unsigned int cycle) const
{

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);


  data_out.add_data_vector(this->solution, "u");

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  const std::string filename =("dg_advection-" +
				Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));

  std::ofstream output((filename + ".vtu"));
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++i)
        filenames.push_back("dg_advection-"+ Utilities::int_to_string(i, 4) + ".vtu");

      std::ofstream master_output( ("dg_advection.pvtu") );
      data_out.write_pvtu_record(master_output, filenames);
    }


}


template <int dim>
void AdvectionProblem<dim>::run()
{
  for (unsigned int cycle = 0; cycle < 4; ++cycle)
    {
	  pcout << "Cycle " << cycle << std::endl;

      if (cycle == 0)
        {
          GridGenerator::subdivided_hyper_cube(triangulation,4);

          triangulation.refine_global(2);

        }
      else
        refine_grid();


      pcout << "Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;

      setup_system();

      pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

      assemble_system();
      solve(solution);

      output_results(cycle);
    }
}


int main(int argc, char *argv[])
{
  try
    {
	  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      AdvectionProblem<3> dgmethod;
      dgmethod.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
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
      std::cerr << std::endl
                << std::endl
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
