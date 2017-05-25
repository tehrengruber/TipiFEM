using TipiFEM.PolytopalMesh
using TipiFEM.Meshes
using TipiFEM.Meshes.Cell
using StaticArrays
using Base.@pure
using TipiFEM.Utils: canonicalize, @generate_sisd
abstract type AbstractFEBasis end

"""
Index of a local degree of freedom
"""
struct LocalDOFIndex{idx} end

"""
Singleton type specifing a basis of a finite element space.

- `basis_type`: Symbol describing the type of basis (i.e. `:Lagrangian`)
- `approx_order`: Polynomial degree up to which the shape functions
- `dim`:

The actual properties of the basis are defined by functions dispatching
on this type (i.e. multiplicity, local_shape_function).
"""
struct FEBasis{basis_type, approx_order} <: AbstractFEBasis end

"number of local shape functions belonging to the interior of a cell"
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"1-node point")::Int       = 1
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"2-node line")::Int        = (order-1)
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"3-node triangle")::Int    = div((order-1)*(order-2), 2)
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"4-node quadrangle")::Int  = (order-1)*(order-1)
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"4-node tetrahedron")::Int = div((order-3)*(order-2)*(order-1), 6)
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"8-node hexahedron")::Int  = (order-1)*(order-1)*(order-1)
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"6-node prism")::Int       = div((order-2)*(order-1)*(order-1), 2)
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"5-node pyramid")::Int     = div((order-2)*(order-1)*(2*order-3), 6)

for (i, node) in [(1, (0, 0)), (2, (0, 1)), (3, (1, 0))]
  @eval @pure function interpolation_node(basis::FEBasis{:Lagrangian, 1}, ::K,
      ::LocalDOFIndex{$(i)}; real_t=Float64) where K <: Cell
    SVector{2, real_t}($(node))
  end
end

# todo: add local shape functions on different domains then the reference triangle
"""
Total number of local shape functions of the reference element `K`.
"""
@pure function number_of_local_shape_functions(basis::FEBasis{:Lagrangian}, ::K) where K <: Cell
  mapreduce(C -> face_count(K, C) * multiplicity(basis, C()), +, skeleton(K))
end

"""
Retrieve i-th local shape function of the reference element of `K` in the given basis.
"""
@inline function local_shape_function{K <: Cell}(basis::FEBasis, ::K, i::LocalDOFIndex)
  x -> local_shape_function(basis, K(), i, x)
end

"""
Retrieve gradient of the i-th local shape function of the reference element
of `K` in the given basis.
"""
@inline function grad_local_shape_function{K <: Cell}(basis::FEBasis, ::K, i::LocalDOFIndex)
  x -> grad_local_shape_function(basis, K(), i, x)
end

"""
Retrieve a vector containing the i-th local shape function of the reference element
of `K` in the given basis in the i-th position.
"""
@pure function local_shape_functions(basis::FEBasis{:Lagrangian}, ::K) where K <: Cell
  [local_shape_function(basis, K(), LocalDOFIndex{i}()) for i in 1:number_of_local_shape_functions(basis, K())]
end

"""
Retrieve a vector containing the i-th gradient of the local shape function of
the reference element of `K` in the given basis in the i-th position.
"""
@pure function grad_local_shape_functions(basis::FEBasis{:Lagrangian}, ::K) where K <: Cell
  # todo: use sizedarray
  [grad_local_shape_function(basis, K(), LocalDOFIndex{i}()) for i in 1:number_of_local_shape_functions(basis, K())]
end

"""
Evaluate all local shape functions at points `x̂s` (local coordinates)
returning a vector of dimension `num_fns` with eltype
`SVector{num_points, T}`. The i-th element of the outer vector is another
vector containing the values of the i-th shape function at `x̂s`.
"""
@generated function local_shape_functions(basis::FEBasis{:Lagrangian, order}, ::K,
    x̂s::SMatrix{num_points, local_dim, T}) where {order, K <: Cell, num_points, local_dim, T <: Real}
  #assert(local_dim == dim(K))
  num_fns=number_of_local_shape_functions(basis(), K())
  expr = Expr(:call, :(SVector{$(num_fns), SVector{num_points, T}}))
  for i in 1:num_fns
    push!(expr.args, quote
      local_shape_function(basis, K(), LocalDOFIndex{$(i)}(), x̂s)
    end)
  end
  expr
end

function local_shape_functions(basis::FEBasis{:Lagrangian, order}, ::K,
    x̂::SVector{local_dim, T}) where {order, K <: Cell, local_dim, T <: Real}
  map(local_shape_functions(basis, K(), convert(SMatrix{1, local_dim, T}, x̂))) do v
    convert(T, v)
  end
end

"""
Evaluate all local shape functions at points `x̂s` (local coordinates)
returning a vector of dimension `num_fns` with eltype
`SVector{num_points, T}`. The i-th element of the outer vector is a
matrix containing the gradients of the i-th shape function horizontally
concatinated.
"""
@generated function grad_local_shape_functions(basis::FEBasis{:Lagrangian, order}, ::K,
    x̂s::SMatrix{num_points, local_dim, T}) where {order, K <: Cell, num_points, local_dim, T <: Real}
  #assert(local_dim == dim(K))
  num_fns=number_of_local_shape_functions(basis(), K())
  expr = Expr(:call, :(SVector{$(num_fns), SMatrix{num_points, local_dim, T}}))
  for i in 1:num_fns
    push!(expr.args, quote
      grad_local_shape_function(basis, K(), LocalDOFIndex{$(i)}(), x̂s)
    end)
  end
  expr
end

function grad_local_shape_functions(basis::FEBasis{:Lagrangian, order}, ::K,
    x̂::SVector{local_dim, T}) where {order, K <: Cell, local_dim, T <: Real}
  map(grad_local_shape_functions(basis, K(), convert(SMatrix{1, local_dim, T}, x̂))) do grad
    convert(SVector{local_dim, T}, grad)
  end
end

#@vectorized_function begin
#  SIMD => function local_shape_function{T, num_points}(
#      ::FEBasis{:Lagrangian, 1}, ::Polytope"3-node triangle", ::LocalDOFIndex{$(i)},
#       x̂::SMatrix{@VectorWidth(num_points), 2, T})
#    $(lsf)
#  end
#  SISD => SIMD[:, 1]
#end

#
# p=1
#
let lsfs = [
      :(1 .- x̂[:, 1] .- x̂[:, 2]),
      :(x̂[:, 1]),
      :(x̂[:, 2])],
    grad_lsfs = [
      (-1, -1),
      (1, 0),
      (0, 1)]
  # given the local shape functions as expressions iterate over them and generate
  #  the corresponding function definitions
  for (i, (lsf, grad_lsf)) in enumerate(zip(lsfs, grad_lsfs))
    @eval @generate_sisd @Base.pure function local_shape_function{T, num_points}(
        ::FEBasis{:Lagrangian, 1}, ::Polytope"3-node triangle", ::LocalDOFIndex{$(i)},
        x̂::SMatrix{@VectorWidth(num_points), 2, T})::SVector{num_points, T}
      $(lsf)
    end
    @eval @generate_sisd @Base.pure function grad_local_shape_function{T, num_points}(
        ::FEBasis{:Lagrangian, 1}, ::Polytope"3-node triangle", ::LocalDOFIndex{$(i)},
         x̂::SMatrix{@VectorWidth(num_points), 2, T})::SMatrix{num_points, 2, T}
      # todo: use repeat iteration instead of copying
      hcat(ones(num_points)*$(grad_lsf[1]), ones(num_points)*$(grad_lsf[2]))
    end
  end
end

let lsfns = [
      :((1 .- x̂[:, 1]) .* (1.-x̂[:, 2])),
      :(x̂[:, 1]        .* (1.-x̂[:, 2])),
      :(x̂[:, 1]        .*     x̂[:, 2]),
      :((1.-x̂[:, 1])   .*     x̂[:, 2])],
    grad_lsfs = [
      (:(-(1-x̂[:, 2])), :(-(1-x̂[:, 1]))),
      (:(  1-x̂[:, 2] ), :(   -x̂[:, 1] )),
      (:(    x̂[:, 2] ), :(    x̂[:, 1] )),
      (:(   -x̂[:, 2] ), :(  1-x̂[:, 1] ))]
  # given the local shape functions as expressions iterate over them and generate
  #  the corresponding function definitions
  for (i, (fn, grad)) in enumerate(zip(lsfns, grad_lsfs))
    @eval @generate_sisd @Base.pure function local_shape_function{T, num_points}(
        ::FEBasis{:Lagrangian, 1}, ::Polytope"4-node quadrangle", ::LocalDOFIndex{$(i)},
        x̂::SMatrix{@VectorWidth(num_points), 2, T})::SVector{num_points, T}
      $(fn)
    end
    @eval @generate_sisd @Base.pure function grad_local_shape_function{T, num_points}(
        ::FEBasis{:Lagrangian, 1}, ::Polytope"4-node quadrangle", ::LocalDOFIndex{$(i)},
         x̂::SMatrix{@VectorWidth(num_points), 2, T})::SMatrix{num_points, 2, T}
      # todo: use repeat iteration instead of copying
      hcat($(grad[1]), $(grad[2]))
    end
  end
end

#
# p=2
#
# todo: use devecortize
let lsfns = [
      :((2.*λ_1(x̂).-1).*λ_1(x̂)),
      :((2.*λ_2(x̂).-1).*λ_2(x̂)),
      :((2.*λ_3(x̂).-1).*λ_3(x̂)),
      :(4.*λ_1(x̂).*λ_2(x̂)),
      :(4.*λ_2(x̂).*λ_3(x̂)),
      :(4.*λ_1(x̂).*λ_3(x̂))
    ],
    grad_lsfns = [
      (:(1.-4.*λ_1(x̂)),  :(1.-4.*λ_1(x̂))),
      (:(4.*λ_2(x̂).-1),  :(0)),
      (:(0),             :(4.*λ_3(x̂).-1)),
      (:(λ_1(x̂)-λ_2(x̂)), :(-λ_2(x̂))),
      (:(λ_3(x̂)),        :(λ_2(x̂))),
      (:(-λ_3(x̂)),       :(λ_1(x̂).-λ_3(x̂)))
    ]
  for (i, (fn, grad)) in enumerate(zip(lsfns, grad_lsfns))
    @eval let λ_1 = local_shape_function(FEBasis{:Lagrangian, 1}(), Polytope"3-node triangle"(), LocalDOFIndex{1}()),
              λ_2 = local_shape_function(FEBasis{:Lagrangian, 1}(), Polytope"3-node triangle"(), LocalDOFIndex{2}()),
              λ_3 = local_shape_function(FEBasis{:Lagrangian, 1}(), Polytope"3-node triangle"(), LocalDOFIndex{3}())
        @generate_sisd @inline @pure function local_shape_function{T, num_points}(
            ::FEBasis{:Lagrangian, 2}, ::Polytope"3-node triangle", ::LocalDOFIndex{$(i)},
             x̂::SMatrix{@VectorWidth(num_points), 2, T})::SVector{num_points, T}
          $(fn)
        end
        @generate_sisd @inline @pure function grad_local_shape_function{T, num_points}(
            ::FEBasis{:Lagrangian, 2}, ::Polytope"3-node triangle", ::LocalDOFIndex{$(i)},
             x̂::SMatrix{@VectorWidth(num_points), 2, T})::SMatrix{num_points, 2, T}
          hcat($(grad[1]), $(grad[2]))
        end
      end
    end
end
