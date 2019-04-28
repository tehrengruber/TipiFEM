using SparseArrays
using TipiFEM.Utils: @sanitycheck

include("febasis.jl")
include("dofhandler.jl")
include("fespace.jl")
include("triplets.jl")
include("norms.jl")

function matrix_assembler(el_matrix::Function, trial_space::FESpace, test_space::FESpace; matrix_type::Type=Triplets)
  # select the active cells of lowest amount from the trial and test space
  # note that the selected active cells must be a subset of the not
  #  selected active cells
  let active_cells = if length(active_cells(trial_space)) < length(active_cells(test_space))
      active_cells(trial_space)
    else
      active_cells(test_space)
    end
    # invoke the actual matrix assembler
    A = matrix_assembler(el_matrix::Function, active_cells, trial_space, test_space)

    if matrix_type == Triplets
      return A
    elseif matrix_type == SparseMatrixCSC
      return sparse(A, number_of_dofs(test_space), number_of_dofs(trial_space))
    else
      error("Invalid argument matrix_type = $(matrix_type)")
    end
  end
end

function matrix_assembler(el_matrix_assembler::Function, cells::HomogeneousIdIterator{C},
    trial_space::FESpace, test_space::FESpace; triplets=nothing) where C <: Cell
  let dofh = dofh(trial_space),
      basis_trial = basis(trial_space),
      basis_test = basis(test_space),
      mesh = mesh(trial_space),
      element_matrix = el_matrix_assembler(basis_trial, basis_test)
    # allocate triplets
    if triplets == nothing
      # approximate number of triplets
      N::Int = number_of_cells(mesh, C())*number_of_local_shape_functions(basis_trial, C())^2
      triplets = Triplets{real_type(mesh)}(N)
    end
    # ensure that the triplets type is inferred
    triplets::Triplets{real_type(mesh)}
    # compute and distribute element matrix for each cell
    for (cid, geo) in graph(geometry(mesh, cells))
      # get global indices for the current element
      dofs = dofh[cid]
      # get active degrees of freedom
      isactive = active_dofs(trial_space, cid)
      # in prior versions dofs belonging to dirichlet nodes in the trial space
      #  were automatically marked inactive. However, we can just disable them
      #@sanitycheck @assert(active_dofs(trial_space, cid) == active_dofs(test_space, cid),
      #  "DOFs on $(cid) that were active in the trial space where not active in the test space")
      # assemble element matrix
      el_mat = element_matrix(cid, geo)
      @sanitycheck @assert length(el_mat) == number_of_local_shape_functions(basis_trial, cell_type(cells)())^2
      # compute and distribute element stiffness matrix
      for i in 1:size(el_mat, 1)
        for j in 1:size(el_mat, 2)
          if isactive[j]
          #if isactive[i] && isactive[j] # agumentation
            push!(triplets, dofs[j], dofs[i], el_mat[i, j])
          end
        end
      end
    end
    triplets
  end
end

function matrix_assembler(el_matrix_assembler::Function, cells::HeterogenousIdIterator, trial_space::FESpace, test_space::FESpace)
  @assert dofh(trial_space).offset==dofh(test_space).offset "dofh of the trial and test space must be equal"
  @assert mesh(trial_space)==mesh(test_space) "finite element spaces must be defined on the same mesh"
  let dofh = dofh(trial_space),
      basis_trial = basis(trial_space),
      basis_test = basis(test_space),
      mesh = mesh(trial_space)
    # total number of local contributions
    N::Int = mapreduce(+, uniontypes(cell_type(cells))) do K
      number_of_cells(mesh, K())*number_of_local_shape_functions(basis_trial, K())^2
    end
    # allocate triplets
    triplets = Triplets{real_type(mesh)}(N)
    # call the assembler on all homogenous id iterators
    foreach(decompose(cells)) do cells
      matrix_assembler(el_matrix_assembler, cells, trial_space, test_space, triplets=triplets)
    end
    triplets
  end
end

function vector_assembler(element_vector_assembler::Function, fespace::FESpace)
  let dofh = dofh(fespace),
      mesh = mesh(fespace),
      basis = basis(fespace),
      mesh_geo = geometry(mesh, active_cells(fespace))
    N = number_of_dofs(fespace)
    V = zeros(real_type(mesh), N)
    k = 1
    foreach(decompose(mesh_geo)) do mesh_geo
      K = cell_type(mesh_geo)
      n = number_of_local_shape_functions(basis, K())
      el_vector = element_vector_assembler(basis)
      for (cid, geo) in graph(mesh_geo[K])
        # get degrees of freedom for the current element
        dofs = dofh[cid]
        # assemble element vector
        el_vec = el_vector(cid, geo)
        # distribute local/element load vector
        for i in 1:n
          V[dofs[i]] += el_vec[i]
        end
      end
    end
    V
  end
end

"""
    incorporate_constraints(fespace::FESpace, args...)

Incorporate constraints imposed on the finite element space `fespace` into
the galerkin matrix and/or rhs vector. Note that this function is just a wrapper
extracting the constraints from the FE space, see the actual implementations of
`incorporate_constraints` dispatching on `IndexMapping`s for more details.

Note that this functions expects all rows of the galerkin matrix belonging to
constrained dofs to be zero, i.e. by marking them inactive in the test space.
"""
function incorporate_constraints!(fespace::FESpace, args...)
  for (key, constraint) in constraints(fespace)
    incorporate_constraints!(constraint, args...)
  end
end

"""
    incorporate_constraints(constraint::IndexMapping, galerkin_matrix::Triplets,
                            rhs_vector::AbstractVector)

Incorporate `constraint`s into galerkin matrix and/or rhs vector, where
`constraint` is an index mapping from dof-indices to the imposed values.
"""
function incorporate_constraints!(constraint::IndexMapping, galerkin_matrix::Triplets, rhs_vector::AbstractVector)
  dof_processed = Set(Int[])
  for (dof, v) in zip(indices(constraint), values(constraint))
    # set diagonal entries of constrained dofs to 1
    #@sanitycheck @assert nnz(galerkin_matrix[dof, :]) == 0 "do not test where the solution is known (at dof: $(dof))"
    if !(dof ∈ dof_processed)
      push!(galerkin_matrix, dof, dof, 1.)
    end
    push!(dof_processed, dof)
    #galerkin_matrix[dof, dof] = 1
    # modify rhs vector
    rhs_vector[dof] = v
  end
end

"""
    incorporate_constraints!(constraint::IndexMapping, galerkin_matrix::Triplets)

Incorporate `constraint`s into the galerkin matrix where `constraint` is an
index mapping from dof-indices to the imposed values.
"""
function incorporate_constraints!(constraint::IndexMapping, galerkin_matrix::Triplets)
  dof_processed = Set(Int[])
  for dof in indices(constraint)
    if !(dof ∈ dof_processed)
      # set all diagonal entries belonging to constraint degrees of freedom to 1
      push!(galerkin_matrix, dof, dof, 1.)
    end
    push!(dof_processed, dof)
  end
end

"""
    incorporate_constraints!(constraint::IndexMapping, rhs_vector::AbstractVector{<:Number, 1})

Incorporate `constraint`s into the rhs vector where `constraint` is an index
mapping from dof-indices to the imposed values.
"""
function incorporate_constraints!(constraint::IndexMapping, rhs_vector::AbstractVector{<:Number})
  for (dof, v) in zip(indices(constraint), values(constraint))
    rhs_vector[dof] = v
  end
end

#function vector_assembler(l::Function, msh::Mesh, basis::FEBasis, dofh::AbstractDofHandler)
#  N = number_of_dofs(dofh)
#  V = zeros(real_type(msh), N)
#  k = 1
#  map(cell_types(msh)) do K
#    n = number_of_local_shape_functions(basis, K())
#    el_vec = element_load_vector(l, K(), basis)
#    for (cidx, geo) in graph(geometry(msh)[K])
#      # get degrees of freedom for the current element
#      dofs = dofh[cidx]
#      # distribute local/element load vector
#      for i in 1:n
#          V[dofs[i]] += el_vec[i](geo)
#      end
#    end
#  end
#  V
#end

#struct Functional
#  args::Array{Symbol, 1}
#  expr::Expr
#  attributes::Dict{Symbol, Any}
#end
#@functional a(u, v) = integrate(α(x) grad(u)(x̂)v(x̂), {x ∈ K})

#{1 < x < 2}

#
#quote
#  K -> [a(b̂[1])]
#end

#a(b̂[i], b̂[j])

#a(u, v) = geo -> integrate_local(x̂->u(x̂)v(x̂), geo)

##
#@functional a(u, v) = ∫ α(x) * u(̂x)v(̂x) dΩ_1
#@functional a(u, v) = ∫ α(x) * u(̂x)v(̂x) d∂Ω
#@functional a(u, v) = ∫ α(x) * u(̂x)v(̂x) dΓ_1
#
#set_attribute(a, :symmetric, true)
#
#a(u, v) = geo -> integrate(x->u(x)*v(x))

#
#@functional l(v) = ∫ v(x) dx
#
#a(basis_fn[i], basis_fn[j])
