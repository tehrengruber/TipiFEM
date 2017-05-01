include("febasis.jl")
include("dofhandler.jl")

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
  N = mapreduce(K -> number_of_cells(msh, K())*number_of_local_shape_functions(basis, K())^2, +, cell_types(msh)) # not used yet, todo: use
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

function vector_assembler(l::Function, msh::Mesh, basis::FEBasis, dofh::AbstractDofHandler)
  N = number_of_dofs(dofh)
  V = Array{real_type(msh), 1}(N)
  k = 1
  map(cell_types(msh)) do K
    n = number_of_local_shape_functions(basis, K())
    el_vec = element_load_vector(l, K(), basis)
    for (cidx, geo) in graph(geometry(msh)[K])
      # get degrees of freedom for the current element
      dofs = dofh[cidx]
      # distribute local/element load vector
      for i in 1:n
          V[dofs[i]] = el_vec[i](geo)
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

#@functional a(u, v) = integrate(α(x) u(x̂)v(x̂), {x ∈ K})

#{1 < x < 2}

#
#quote
#  K -> [a(b̂[1])]
#end

#a(b̂[i], b̂[j])

#a(u, v) = geo -> integrate_local(x̂->u(x̂)v(x̂), geo)
##
#@functional a(u, v) = ∫ α(x) * u(̂x)v(̂x) dK̂
#
#set_attribute(a, :symmetric, true)
#
#a(u, v) = geo -> integrate(x->u(x)*v(x))

#
#@functional l(v) = ∫ v(x) dx
#
#a(basis_fn[i], basis_fn[j])
