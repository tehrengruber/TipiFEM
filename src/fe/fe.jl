using TipiFEM.Utils.@sanitycheck

include("febasis.jl")
include("dofhandler.jl")
include("fespace.jl")

# local assembler
"""
Assemble local element matrix
"""
function element_stiffness_matrix(a::Function, ::K, basis::FEBasis) where K <: Cell
  n=number_of_local_shape_functions(basis, K())
  el_matrix = reshape(Array{Function, 2}(n, n), Size(n, n))
  basis_fn=local_shape_functions(basis, K())
  for i in 1:n
    for j in 1:n
      #el_matrix[i, j] = a(basis_fn[i], basis_fn[j])
      el_matrix[i, j] = a(i, j)
    end
  end
  el_matrix
end

"""
Assemble local load vector
"""
function element_load_vector(l::Function, ::K, basis::FEBasis) where K <: Cell
  n=number_of_local_shape_functions(basis, K())
  el_vec = reshape(Array{Function, 1}(n), Size(n))
  basis_fn=local_shape_functions(basis, K())
  for i in 1:n
    el_vec[i] = l(i)
  end
  el_vec
end

"""
Assemble global stiffness matrix
"""
function matrix_assembler(a::Function, msh::Mesh, basis::FEBasis, dofh::AbstractDofHandler)
  N = mapreduce(K -> number_of_cells(msh, K())*number_of_local_shape_functions(basis, K())^2, +, cell_types(msh))
  I = Array{Int, 1}(N)
  J = Array{Int, 1}(N)
  V = Array{real_type(msh), 1}(N)
  k = 1
  map(cell_types(msh)) do K
    n = number_of_local_shape_functions(basis, K())
    el_matrix = element_stiffness_matrix(a, K(), basis)
    for (cidx, geo) in graph(geometry(msh)[K])
      # get degrees of freedom for the current element
      dofs = dofh[cidx]
      # distribute local/element stiffness matrix matrix
      for i in 1:n
        for j in 1:n
          # todo: double check "swapped" indices
          I[k] = dofs[j]
          J[k] = dofs[i]
          V[k] = el_matrix[i, j](geo)
          k+=1
        end
      end
    end
  end
  sparse(I, J, V)
end

function matrix_assembler(el_matrix::Function, trial_space::FESpace, test_space::FESpace)
  @assert dofh(trial_space)==dofh(test_space) "dofh of the trial and test space must be equal"
  # choose the mesh with the lowest number of elements
  let mesh = number_of_elements(mesh(trial_space)) > number_of_elements(mesh(test_space)),
      dofh = dofh(trial_space),
      basis_trial = basis(trial_space),
      basis_test = basis(test_space),
      mesh_geo = geometry(mesh(trial_space))

    # todo: sanity checks on a
    # total number of local contributions
    N = min(number_of_active_dofs(trial_space), number_of_active_dofs(test_space))
    # row indices
    I = Array{Int, 1}(N)
    # column indices
    J = Array{Int, 1}(N)
    # matrix values
    V = Array{real_type(mesh), 1}(N)
    # current index into the row-indices, column-indices, values array
    k = 1
    # compute matrix elements for each cell type
    map(cell_types(mesh)) do K
      # compute and distribute element matrix for each cell
      for (cidx, geo) in graph(mesh_geo[K]) # todo: remove [K]
        # get global indices for the current element
        dofs = dofh[cidx]
        # get active degrees of freedom
        isactive = active_dofs(trial_space, cidx)
        @sanitycheck assert(active_dofs(trial_space, cidx) == active_dofs(test_space, cidx))
        # asseble element matrix
        el_mat = el_matrix(cidx, geo, dofs)
        # compute and distribute element stiffness matrix
        for i in 1:size(el_mat, 1)
          for j in 1:size(el_mat, 2)
            if is_dof_active(trial_space, dofs[i]) && is_dof_active(trial_space, dofs[j])
              I[k] = dofs[j]
              J[k] = dofs[i]
              V[k] = el_mat[i, j]
              k+=1
            end
          end
        end
      end
    end
    # convert triplets into CSC format
    m = sparse(I, J, V)
    # remove rows of inactive cells in the test space
    #  since this is CSC this might be expensive
    #m[inactive_cells(test_space), :] = 0
    # remove coloums of inactive cells in the trial space
    #m[:, inactive_cells(trial_space)] = 0
  end
end

"incorporate constraints into the galerkin matrix and rhs vector"
function incorporate_constraints(fespace, galerkin_matrix, rhs_vector)
  for constraint in constraints(trial_space)
    for (id, v) in graph(constraint)
      # set diagonal entries of constrained dofs to 1
      @sanitycheck @assert m[id, :] == 0 "do not test where the solution is known"
      m[id, id] = 1
      # modify rhs vector
      rhs_vector[id] = v
    end
  end
end

function vector_assembler(l::Function, msh::Mesh, basis::FEBasis, dofh::AbstractDofHandler)
  N = number_of_dofs(dofh)
  V = zeros(real_type(msh), N)
  k = 1
  map(cell_types(msh)) do K
    n = number_of_local_shape_functions(basis, K())
    el_vec = element_load_vector(l, K(), basis)
    for (cidx, geo) in graph(geometry(msh)[K])
      # get degrees of freedom for the current element
      dofs = dofh[cidx]
      # distribute local/element load vector
      for i in 1:n
          V[dofs[i]] += el_vec[i](geo)
      end
    end
  end
  V
end

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

a(u, v) = geo -> integrate_local(x̂->u(x̂)v(x̂), geo)

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
