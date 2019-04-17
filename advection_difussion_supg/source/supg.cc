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



#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
//#include <deal.II/lac/solver_cg.h>
//#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>


#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include <math.h>

using namespace dealii;



class Step3
{
public:
  Step3();

  void run();


private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<2> triangulation;
  FE_Q<2>          fe;
  DoFHandler<2>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};


Step3::Step3()
  : fe(1)
  , dof_handler(triangulation)
{}



void Step3::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(5);

  GridTools::distort_random(0.3, triangulation);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}




void Step3::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



void Step3::assemble_system()
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

  Point<2> velocity;
  velocity(0) = pow(2.0,0.5)/2.0;
  velocity(1) = velocity(0);
  const double speed = 100;
  velocity(0) *= speed;
  velocity(1) *= speed;
  const double nu = 1.0;

  //auto mapping = fe_values.get_mapping();

  for (const auto &cell : dof_handler.active_cell_iterators())
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

      //
      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j){
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) *
                 fe_values.shape_grad(j, q_index) *
                 fe_values.JxW(q_index));

              cell_matrix(i, j) +=
                (velocity * fe_values.shape_grad(i, q_index) *
                 fe_values.shape_value(j, q_index) *
                 fe_values.JxW(q_index));

              cell_matrix(i,j) += (fe_values.shape_grad(i,q_index)*
            		  velocity)*tau*(fe_values.shape_grad(j,q_index)*velocity) *
					  fe_values.JxW(q_index);

            }
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                            1.0 * fe_values.JxW(q_index));
            cell_rhs(i) += (fe_values.shape_grad(i,q_index)*velocity)*tau*fe_values.JxW(q_index);
          }
        }
      cell->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }


  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}



void Step3::solve()
{
//  SolverControl solver_control(1000, 1e-12);
//  SolverCG<> solver(solver_control);

//  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

	SparseDirectUMFPACK solver;

	solver.initialize(system_matrix);

	solver.solve(system_rhs);

}



void Step3::output_results() const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(system_rhs, "solution");
  data_out.build_patches();

  std::ofstream outputgpl("solution3.gpl");
  data_out.write_gnuplot(outputgpl);

  std::ofstream output("solution3.vtk");
  data_out.write_vtk(output);

}



void Step3::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}



int main()
{
  deallog.depth_console(2);

  Step3 laplace_problem;
  laplace_problem.run();

  return 0;
}
