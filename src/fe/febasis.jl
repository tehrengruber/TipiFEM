using TipiFEM.PolytopalMesh
using TipiFEM.Meshes
using TipiFEM.Meshes.Cell
using StaticArrays
using Base.@pure
using TipiFEM.Utils: canonicalize, @generate_sisd
abstract type AbstractFEBasis end

struct FEBasis{basis_type, approx_order} <: AbstractFEBasis end

"number of local shape functions belonging to subentities of a particular type"
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"1-node point")::Int       = 1;
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"2-node line")::Int        = (order-1);
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"3-node triangle")::Int    = (order-1)*(order-2)/2;
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"4-node quadrangle")::Int  = (order-1)*(order-1);
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"4-node tetrahedron")::Int = (order-3)*(order-2)*(order-1)/6;
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"8-node hexahedron")::Int  = (order-1)*(order-1)*(order-1);
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"6-node prism")::Int       = (order-2)*(order-1)*(order-1)/2;
multiplicity{order}(::FEBasis{:Lagrangian, order}, ::Polytope"5-node pyramid")::Int     = (order-2)*(order-1)*(2*order-3)/6

# todo: add local shape functions on different domains then the reference triangle
@pure function number_of_local_shape_functions(basis::FEBasis{:Lagrangian}, ::K) where K <: Cell
  mapreduce(C -> face_count(K, C) * multiplicity(basis, C()), +, skeleton(K))
end

@inline function local_shape_function{K <: Cell}(basis::FEBasis{:Lagrangian}, ::K, lidx::LocalIndex)
  x -> local_shape_function(basis, K(), lidx, x)
end

@inline function grad_local_shape_function{K <: Cell}(basis::FEBasis{:Lagrangian}, ::K, lidx::LocalIndex)
  x -> grad_local_shape_function(basis, K(), lidx, x)
end

@pure function local_shape_functions(basis::FEBasis{:Lagrangian}, ::K) where K <: Cell
  [local_shape_function(basis, K(), LocalIndex{i}()) for i in 1:number_of_local_shape_functions(basis, K())]
end

@pure function grad_local_shape_functions(basis::FEBasis{:Lagrangian}, ::K) where K <: Cell
  # todo: use sizedarray
  [grad_local_shape_function(basis, K(), LocalIndex{i}()) for i in 1:number_of_local_shape_functions(basis, K())]
end

"""
Evaluate all local shape functions at points `x̂s` (local coordinates)
returning a vector of dimension `num_fns` with eltype
`SVector{num_points, T}`. The i-th element of the outer vector is a
vector containing the values of the i-th shape function at `x̂s`.
"""
@generated function local_shape_functions(basis::FEBasis{:Lagrangian, order}, ::K,
    x̂s::SMatrix{num_points, local_dim, T}) where {order, K <: Cell, num_points, local_dim, T <: Real}
  #assert(local_dim == dim(K))
  num_fns=number_of_local_shape_functions(basis(), K())
  expr = Expr(:call, :(SVector{$(num_fns), SVector{num_points, T}}))
  for i in 1:num_fns
    push!(expr.args, quote
      local_shape_function(basis, K(), LocalIndex{$(i)}(), x̂s)
    end)
  end
  expr
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
      grad_local_shape_function(basis, K(), LocalIndex{$(i)}(), x̂s)
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
#      ::FEBasis{:Lagrangian, 1}, ::Polytope"3-node triangle", ::LocalIndex{$(i)},
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
        ::FEBasis{:Lagrangian, 1}, ::Polytope"3-node triangle", ::LocalIndex{$(i)},
        x̂::SMatrix{@VectorWidth(num_points), 2, T})::SVector{num_points, T}
      $(lsf)
    end
    @eval @generate_sisd @Base.pure function grad_local_shape_function{T, num_points}(
        ::FEBasis{:Lagrangian, 1}, ::Polytope"3-node triangle", ::LocalIndex{$(i)},
         x̂::SMatrix{@VectorWidth(num_points), 2, T})::SMatrix{num_points, 2, T}
      # todo: use repeat iteration instead of copying
      hcat(ones(num_points)*$(grad_lsf[1]), ones(num_points)*$(grad_lsf[2]))
    end
  end
end

let lsfs = [
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
  for (i, (lsf, grad_lsf)) in enumerate(zip(lsfs, grad_lsfs))
    @eval @generate_sisd @Base.pure function local_shape_function{T, num_points}(
        ::FEBasis{:Lagrangian, 1}, ::Polytope"4-node quadrangle", ::LocalIndex{$(i)},
        x̂::SMatrix{@VectorWidth(num_points), 2, T})::SVector{num_points, T}
      $(lsf)
    end
    @eval @generate_sisd @Base.pure function grad_local_shape_function{T, num_points}(
        ::FEBasis{:Lagrangian, 1}, ::Polytope"4-node quadrangle", ::LocalIndex{$(i)},
         x̂::SMatrix{@VectorWidth(num_points), 2, T})::SMatrix{num_points, 2, T}
      # todo: use repeat iteration instead of copying
      hcat($(grad_lsf[1]), $(grad_lsf[2]))
    end
  end
end

#
# p=2
#
# todo: use devecortize
let λ_1 = local_shape_function(FEBasis{:Lagrangian, 1}(), Polytope"3-node triangle"(), LocalIndex{1}()),
    λ_2 = local_shape_function(FEBasis{:Lagrangian, 1}(), Polytope"3-node triangle"(), LocalIndex{2}()),
    λ_3 = local_shape_function(FEBasis{:Lagrangian, 1}(), Polytope"3-node triangle"(), LocalIndex{3}()),
    lsf = [
        :((2.*λ_1(x̂).-1).*λ_1(x̂)),
        :((2.*λ_2(x̂).-1).*λ_2(x̂)),
        :((2.*λ_3(x̂).-1).*λ_3(x̂)),
        :(4.*λ_1(x̂).*λ_2(x̂)),
        :(4.*λ_2(x̂).*λ_3(x̂)),
        :(4.*λ_1(x̂).*λ_3(x̂))
    ]
    for (i, fnexpr) in enumerate(lsf)
      @eval @generate_sisd @inline @pure function local_shape_function{T, num_points}(
          ::FEBasis{:Lagrangian, 2}, ::Polytope"3-node triangle", ::LocalIndex{$(i)},
           x̂::SMatrix{@VectorWidth(num_points), 2, T})::SVector{num_points, T}
        $(fnexpr)
      end
      # todo: gradients
    end
end
