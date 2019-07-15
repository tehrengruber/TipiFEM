using TipiFEM.PolytopalMesh
using TipiFEM.Meshes
using TipiFEM.Meshes: Cell
using StaticArrays
using Base: @pure
using TipiFEM.Utils: canonicalize, @generate_sisd
abstract type AbstractFEBasis end

"""
Local degree of freedom
"""
struct LocalDOF{basis, K <: Cell, idx} end

#"""
#Degree of freedom
#"""
#struct DOF{basis, C <: Cell}
#  cid::Id{C}
#  lidx::Int
#end

"""
Local interpolation node
"""
struct LocalInterpolationNode{basis, K <: Cell , idx} end

"""
Global index of an interpolation node
"""
struct InterpolationNodeIndex{C <: Cell}
  cid::Id{C}
  lidx::Int
end

struct InterpolationNode{world_dim, REAL_T <: Real}
  idx::InterpolationNodeIndex
  coords::SVector{world_dim, REAL_T}

  function InterpolationNode(idx::InterpolationNodeIndex, coords::SVector{world_dim, REAL_T}) where {world_dim, REAL_T <: Real}
    new{world_dim, REAL_T}(idx, coords)
  end
end

"""
Singleton type specifing a basis of a finite element space.

- `basis_type`: Symbol describing the type of basis (e.g. `:Lagrangian`)
- `approx_order`: Polynomial degree

The actual properties of the basis are defined by functions dispatching
on this type (i.e. multiplicity, local_shape_function).
"""
struct FEBasis{basis_type, approx_order} <: AbstractFEBasis end

################################################################################
# multiplicity
################################################################################
# todo: check that all return types are inferred correctly to int
"number of local shape functions belonging to the interior of a cell"
multiplicity(::FEBasis{:Lagrangian, order}, ::Polytope"1-node point") where order       = 1
multiplicity(::FEBasis{:Lagrangian, order}, ::Polytope"2-node line") where order        = (order-1)
multiplicity(::FEBasis{:Lagrangian, order}, ::Polytope"3-node triangle") where order    = (div((order-1)*(order-2), 2))
multiplicity(::FEBasis{:Lagrangian, order}, ::Polytope"4-node quadrangle") where order  = (order-1)*(order-1)
multiplicity(::FEBasis{:Lagrangian, order}, ::Polytope"4-node tetrahedron") where order = div((order-3)*(order-2)*(order-1), 6)
multiplicity(::FEBasis{:Lagrangian, order}, ::Polytope"8-node hexahedron") where order  = (order-1)*(order-1)*(order-1)
multiplicity(::FEBasis{:Lagrangian, order}, ::Polytope"6-node prism") where order       = div((order-2)*(order-1)*(order-1), 2)
multiplicity(::FEBasis{:Lagrangian, order}, ::Polytope"5-node pyramid") where order     = div((order-2)*(order-1)*(2*order-3), 6)

################################################################################
# local coordinates of the interpolation nodes
################################################################################
let all_coords = Dict(
    # linear lagrangian finite elements on edges
    (Polytope"2-node line", 1) => ((0,), (1,)),
    # second order lagrangian finite elements on edges
    (Polytope"2-node line", 2) => ((0,), (1,), (0.5,)),
    # linear lagrangian finite elements on triangles
    (Polytope"3-node triangle", 1) => ((0, 0), (1, 0), (0, 1)),
    # second order lagrangian finite elements on triangles
    (Polytope"3-node triangle", 2) => ((0, 0), (1, 0), (0, 1), (0.5, 0), (0.5, 0.5), (0, 0.5)),
    # linear lagrangian finite elements on quadrilaterals
    (Polytope"4-node quadrangle", 1) => ((0, 0), (1, 0), (1, 1), (0, 1)),
    # second order lagrangian finite elements on quadrilaterals
    (Polytope"4-node quadrangle", 2) => ((0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5), (0.5, 0.5)))
  # generate `coordinates` functions
  for ((K, order), coords) in all_coords
    for (i, coord) in enumerate(coords)
      @eval @pure function coordinates(::LocalInterpolationNode{FEBasis{:Lagrangian, $(order)}, $(K), $(i)},
          ::Type{T}=Float64) where T <: Number
        SVector{$(length(coord)), T}($(coord))
      end
    end
  end
end

################################################################################
# local cell indices of cells belonging to local interpolation nodes
################################################################################
let attached_cells = Dict(
    # linear lagrangian finite elements on triangles
    (Polytope"2-node line", 1) => ((Polytope"1-node point" for i in 1:3)...,),
    # second order lagrangian finite elements on triangles
    (Polytope"2-node line", 2) => ((Polytope"1-node point" for i in 1:3)...,
                                    Polytope"2-node line"),
    # linear lagrangian finite elements on triangles
    (Polytope"3-node triangle", 1) => ((Polytope"1-node point" for i in 1:3)...,),
    # second order lagrangian finite elements on triangles
    (Polytope"3-node triangle", 2) => ((Polytope"1-node point" for i in 1:3)...,
                                       (Polytope"2-node line" for i in 1:3)...,),
    # linear lagrangian finite elements on quadrilaterals
    (Polytope"4-node quadrangle", 1) => ((Polytope"1-node point" for i in 1:4)...,),
    # second order lagrangian finite elements on quadrilaterals
    (Polytope"4-node quadrangle", 2) => ((Polytope"1-node point" for i in 1:4)...,
                                         (Polytope"2-node line" for i in 1:4)...,
                                          Polytope"4-node quadrangle"))
  for ((K, order), Cs) in attached_cells
    for (i, C) in enumerate(Cs)
      @eval attached_cell(::LocalInterpolationNode{FEBasis{:Lagrangian, $(order)}, $(K), $(i)}) = $(C)
      @eval attached_cell(::LocalDOF{FEBasis{:Lagrangian, $(order)}, $(K), $(i)}) = $(C)
    end
  end
end

################################################################################
# degrees of freedom attached to an interpolation node on the boundary of a cell
################################################################################
let boundary_dofs = Dict(
    (Polytope"1-node point", 1) => (),
    (Polytope"1-node point", 2) => (),
    (Polytope"2-node line", 1)  => (1, 2),
    (Polytope"2-node line", 2)  => (1, 2),
    # linear lagrangian finite elements on triangles
    (Polytope"3-node triangle", 1) => (1, 2, 3),
    # second order lagrangian finite elements on triangles
    (Polytope"3-node triangle", 2) => (1, 2, 3, 4, 5, 6),
    # linear lagrangian finite elements on quadrilaterals
    (Polytope"4-node quadrangle", 1) => (1, 2, 3, 4),
    # second order lagrangian finite elements on quadrilaterals
    (Polytope"4-node quadrangle", 2) => (1, 2, 3, 4, 5, 6, 7, 8))
  for ((K, order), dof_indices) in boundary_dofs
    @eval function boundary_dofs(::FEBasis{:Lagrangian, $(order)}, ::$(K))
      $(map(i->LocalDOF{FEBasis{:Lagrangian, order}, K, i}(), dof_indices))
    end
  end
end

################################################################################
# degrees of freedom attached to an interpolation node on the interior of a cell
################################################################################
let internal_dofs = Dict(
    (Polytope"1-node point", 1) => (1,),
    (Polytope"1-node point", 2) => (1,),
    (Polytope"2-node line", 1)  => (),
    (Polytope"2-node line", 2)  => (3,),
    # linear lagrangian finite elements on triangles
    (Polytope"3-node triangle", 1) => (),
    # second order lagrangian finite elements on triangles
    (Polytope"3-node triangle", 2) => (),
    # linear lagrangian finite elements on quadrilaterals
    (Polytope"4-node quadrangle", 1) => (),
    # second order lagrangian finite elements on quadrilaterals
    (Polytope"4-node quadrangle", 2) => (9,))
  for ((K, order), dof_indices) in internal_dofs
    @eval function internal_dofs(::FEBasis{:Lagrangian, $(order)}, ::$(K))
      $(map(i->LocalDOF{FEBasis{:Lagrangian, order}, K, i}(), dof_indices))
    end
  end
end

################################################################################
# local index of the face to which a boundary dof is attached to
################################################################################
let local_face_indices = Dict(
    # linear lagrangian finite elements on line segments
    (Polytope"2-node line", 1) => (1, 2), # (vertices...)
    # linear lagrangian finite elements on triangles
    (Polytope"3-node triangle", 1) => (1, 2, 3), # (vertices...)
    # second order lagrangian finite elements on triangles
    (Polytope"3-node triangle", 2) => (1, 2, 3, 1, 2, 3), #(vertices..., edges...)
    # linear lagrangian finite elements on quadrilaterals
    (Polytope"4-node quadrangle", 1) => (1, 2, 3, 4), # (vertices...)
    # second order lagrangian finite elements on quadrilaterals
    (Polytope"4-node quadrangle", 2) => (1, 2, 3, 4, 1, 2, 3, 4)) # (vertices..., edges....)
  for ((K, order), face_indices) in local_face_indices
    for (i, face_index) in enumerate(face_indices)
      @eval attached_face(::LocalDOF{FEBasis{:Lagrangian, $(order)}, $(K), $(i)}) = $(face_index)
    end
  end
end

################################################################################
# cell type constrained interpolation node index
################################################################################
#let cell_type_constrained_interpolation_node_indices = Dict(
#    # linear lagrangian finite elements on triangles
#    (Polytope"3-node triangle", 1) => (1, 2, 3), # (vertices...)
#    # second order lagrangian finite elements on triangles
#    (Polytope"3-node triangle", 2) => (1, 2, 3, 1, 2, 3), #(vertices..., edges...)
#    # linear lagrangian finite elements on quadrilaterals
#    (Polytope"4-node quadrangle", 1) => (1, 2, 3, 4), # (vertices...)
#    # second order lagrangian finite elements on quadrilaterals
#    (Polytope"4-node quadrangle", 2) => (1, 2, 3, 4, 1, 2, 3, 4)) # (vertices..., edges....)
#  for ((K, order), face_indices) in local_face_indices
#    for (i, face_index) in enumerate(face_indices)
#      @eval attached_face(::LocalDOF{FEBasis{:Lagrangian, $(order)}, $(K), $(i)}) = $(face_index)
#    end
#  end
#end

"get the index of a local degree of freedom"
local_index(::LocalDOF{B, K, idx}) where {B <: FEBasis, K <: Cell, idx} = idx

#index(fespace, cid, ldof::LocalDOF{B, K, idx}) where {B <: FEBasis, K <: Cell, idx} = dofh(fespace)[cid, local_index(ldof)]

"get the local index of the local interpolation node"
local_index(::LocalInterpolationNode{B, K, idx}) where {B <: FEBasis, K <: Cell, idx} = idx

"get the global index of the local interpolation node"
index(cid::Id{C}, interp_node::LocalInterpolationNode{<:FEBasis, C}) where C <: Cell = InterpolationNodeIndex(cid, local_index(interp_node))

# todo: generailize for multidimensional u's
@pure function interpolation_node(::LocalDOF{FEBasis{:Lagrangian, order}, K, i}) where {order, K <: Cell, i}
  LocalInterpolationNode{FEBasis{:Lagrangian, order}, K, i}()
end

function number_of_interpolation_nodes(basis::FEBasis{:Lagrangian}, ::K) where K <: Cell
  number_of_local_shape_functions(basis, K())
end

"""
Total number of local shape functions of `K`.
"""
@pure function _number_of_local_shape_functions(basis::FEBasis{:Lagrangian}, ::K) where K <: Cell
  mapreduce(C -> face_count(K, C) * multiplicity(basis, C()), +, skeleton(K))
end

@pure function number_of_local_shape_functions(basis::FEBasis{:Lagrangian}, ::K) where K <: Cell
  _number_of_local_shape_functions(basis, K())
end

@generated function interpolation_nodes(basis::FEBasis{:Lagrangian, order}, ::K) where {order, K <: Cell}
  expr = Expr(:tuple)
  for i in 1:number_of_interpolation_nodes(FEBasis{:Lagrangian, order}(), K())
    push!(expr.args, LocalInterpolationNode{FEBasis{:Lagrangian, order}, K, i}())
  end
  expr
end

"""
Retrieve the local shape function attached to the local degree of freedom `i`.
"""
@inline function local_shape_function(i::LocalDOF)
  x -> local_shape_function(i, x)
end

"""
Retrieve gradient of the i-th local shape function of the reference element
of `K` in the given basis.
"""
@inline function grad_local_shape_function(i::LocalDOF)
  x -> grad_local_shape_function(i, x)
end

"""
Retrieve a vector containing the i-th local shape function of the reference element
of `K` in the given basis in the i-th position.
"""
@pure function local_shape_functions(basis::FEBasis{:Lagrangian}, ::K) where K <: Cell
  [local_shape_function(LocalDOF{basis, K, i}()) for i in 1:number_of_local_shape_functions(basis, K())]
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
      local_shape_function(LocalDOF{$(basis), $(K), $(i)}(), x̂s)
    end)
  end
  expr
end

function local_shape_functions(basis::FEBasis{:Lagrangian, order}, ::K,
    x̂::SVector{local_dim, T}) where {order, K <: Cell, local_dim, T <: Real}
  map(local_shape_functions(basis, K(), convert(SMatrix{1, local_dim, T}, x̂))) do v
    v[1]
  end
end

@generated function grad_local_shape_functions(basis::FEBasis{:Lagrangian, order}, ::K,
    x̂s::SMatrix{num_points, local_dim, T}) where {order, K <: Cell, num_points, local_dim, T <: Real}
  #assert(local_dim == dim(K))
  num_fns=number_of_local_shape_functions(basis(), K())
  expr = Expr(:call, :(SVector{$(num_fns), SMatrix{num_points, local_dim, T, $(num_points*local_dim)}}))
  for i in 1:num_fns
    push!(expr.args, quote
      grad_local_shape_function($(LocalDOF{basis, K, i}()), x̂s)
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
#      ::FEBasis{:Lagrangian, 1}, ::Polytope"3-node triangle", ::LocalDOF{$(i)},
#       x̂::SMatrix{@VectorWidth(num_points), 2, T})
#    $(lsf)
#  end
#  SISD => SIMD[:, 1]
#end

#
# p=1
#
let lsfs = [:(1 .- x̂), :(x̂)], grad_lsfs = [-1, 1]
  # given the local shape functions as expressions iterate over them and generate
  #  the corresponding function definitions
  for (i, (lsf, grad_lsf)) in enumerate(zip(lsfs, grad_lsfs))
    @eval @pure function local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"2-node line", $(i)},
        x̂::Union{SVector{num_points, T}, SMatrix{num_points, 1, T}}) where {T, num_points}
      $(lsf)
    end
    @eval @pure function local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"2-node line", $(i)}, x̂::T) where T <: Number
      $(lsf)
    end
    @eval @pure function grad_local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"2-node line", $(i)},
         x̂::Union{SVector{num_points, T}, SMatrix{num_points, 1, T}}) where {T, num_points}
      $(grad_lsf)*ones(SVector{num_points, T})
    end
    @eval @pure function grad_local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"2-node line", $(i)},
         x̂::T) where {T <: Number}
      $(grad_lsf)
    end
  end
end

function local_shape_functions(::FEBasis{:Lagrangian, order}, ::Polytope"2-node line", x̂::T) where {order, T <: Number}
  local_shape_functions(FEBasis{:Lagrangian, order}(), Polytope"2-node line"(), SVector{1, T}(x̂))
end

# triangles
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
    @eval @generate_sisd @pure function local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"3-node triangle", $(i)},
        x̂::SMatrix{@VectorWidth(num_points), 2, T})::SVector{num_points, T} where {T, num_points}
      $(lsf)
    end
    @eval @generate_sisd @pure function grad_local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"3-node triangle", $(i)},
         x̂::SMatrix{@VectorWidth(num_points), 2, T})::SMatrix{num_points, 2, T} where {T, num_points}
      # todo: use repeat iteration instead of copying
      # todo use fixed size ones
      hcat(ones(SVector{num_points, T})*$(grad_lsf[1]), ones(SVector{num_points, T})*$(grad_lsf[2]))
    end
  end
end

# quadrangles
let lsfns = [
      :((1 .- x̂[:, 1]) .* (1 .- x̂[:, 2])),
      :(x̂[:, 1]        .* (1 .- x̂[:, 2])),
      :(x̂[:, 1]        .*     x̂[:, 2]),
      :((1 .- x̂[:, 1])   .*     x̂[:, 2])],
    grad_lsfs = [
      (:(-(1-x̂[:, 2])), :(-(1-x̂[:, 1]))),
      (:(  1-x̂[:, 2] ), :(   -x̂[:, 1] )),
      (:(    x̂[:, 2] ), :(    x̂[:, 1] )),
      (:(   -x̂[:, 2] ), :(  1-x̂[:, 1] ))]
  # given the local shape functions as expressions iterate over them and generate
  #  the corresponding function definitions
  for (i, (fn, grad)) in enumerate(zip(lsfns, grad_lsfs))
    @eval @generate_sisd @pure function local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"4-node quadrangle", $(i)},
        x̂::SMatrix{@VectorWidth(num_points), 2, T})::SVector{num_points, T} where {T, num_points}
      $(fn)
    end
    @eval @generate_sisd @pure function grad_local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"4-node quadrangle", $(i)},
         x̂::SMatrix{@VectorWidth(num_points), 2, T})::SMatrix{num_points, 2, T} where {T, num_points}
      # todo: use repeat iteration instead of copying
      hcat($(grad[1]), $(grad[2]))
    end
  end
end

#
# p=2
#
# todo: use devecortize
# edges
let lsfs = [:(2*(x̂.-1).*(x̂.-0.5)), :(2*x̂.*(x̂ .- 0.5)), :(4*(1 .- x̂).*x̂)],
    grad_lsfs = [:(4x̂-3), :(4x̂-1), :(4-8x̂)]
  # given the local shape functions as expressions iterate over them and generate
  #  the corresponding function definitions
  for (i, (lsf, grad_lsf)) in enumerate(zip(lsfs, grad_lsfs))
    @eval @pure function local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"2-node line", $(i)},
        x̂::Union{SVector{num_points, T}, SMatrix{num_points, 1, T}}) where {T, num_points}
      $(lsf)
    end
    @eval @pure function local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"2-node line", $(i)}, x̂::T) where T <: Number
      $(lsf)
    end
    @eval @pure function grad_local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"2-node line", $(i)},
         x̂::Union{SVector{num_points, T}, SMatrix{num_points, 1, T}}) where {T, num_points}
      $(grad_lsf)*ones(SVector{num_points, T})
    end
    @eval @pure function grad_local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"2-node line", $(i)},
         x̂::T) where T <: Number
      $(grad_lsf)
    end
  end
end

# triangles
let lsfns = [
      :((2 .* λ_1(x̂) .- 1) .* λ_1(x̂)),
      :((2 .* λ_2(x̂) .- 1) .* λ_2(x̂)),
      :((2 .* λ_3(x̂) .- 1) .* λ_3(x̂)),
      :(4 .* λ_1(x̂) .* λ_2(x̂)),
      :(4 .* λ_2(x̂) .* λ_3(x̂)),
      :(4 .* λ_1(x̂) .* λ_3(x̂))
    ],
    grad_lsfns = [
      (:(4x̂[:, 1] .+ 4x̂[:, 2].-3),         :( 4x̂[:, 1] .+ 4x̂[:, 2] .- 3)),
      (:(4x̂[:, 1] .- 1),                   :( zeros(SVector{num_points, T}))),
      (:(zeros(SVector{num_points, T})), :( 4x̂[:, 2]-1)),
      (:(4 .- 8x̂[:, 1] .- 4x̂[:, 2]),         :(-4x̂[:, 1])),
      (:(4x̂[:, 2]),                      :( 4x̂[:, 1])),
      (:(-4x̂[:, 2]),                     :( 4 .- 4x̂[:, 1] .- 8x̂[:, 2]))
      #(:(1.-4.*λ_1(x̂)),  :(1.-4.*λ_1(x̂))),
      #(:(4.*λ_2(x̂).-1),  :(0)),
      #(:(0),             :(4.*λ_3(x̂).-1)),
      #(:(4.*(λ_1(x̂)-λ_2(x̂))), :(-4.*λ_2(x̂))),
      #(:(4.*(λ_3(x̂))),        :(4.*λ_2(x̂))),
      #(:(4.*(-λ_3(x̂))),       :(4.*(λ_1(x̂).-λ_3(x̂))))
    ]

  for (i, (fn, grad)) in enumerate(zip(lsfns, grad_lsfns))
    let λ_1 = local_shape_function(LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"3-node triangle", 1}()),
        λ_2 = local_shape_function(LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"3-node triangle", 2}()),
        λ_3 = local_shape_function(LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"3-node triangle", 3}())
        @eval @generate_sisd @inline @pure function local_shape_function(
            ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"3-node triangle", $(i)},
             x̂::SMatrix{@VectorWidth(num_points), 2, T})::SVector{num_points, T} where {T, num_points}
          $(fn)
        end
        @eval @generate_sisd @inline @pure function grad_local_shape_function(
            ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"3-node triangle", $(i)},
             x̂::SMatrix{@VectorWidth(num_points), 2, T})::SMatrix{num_points, 2, T} where {T, num_points}
          hcat($(grad[1]), $(grad[2]))
        end
      end
    end
end

# quadrilaterals
# 8 interpolation nodes, not used
# tmp1 = x̂[:, 1] .- 1
# tmp2 = x̂[:, 2] .- 1
# :(-tmp1    .* tmp2    .* ( -1 .+ 2 .* x̂[:, 1] + 2 .* x̂[:, 2])),
# :(-tmp2    .* x̂[:, 1] .* ( -1 .+ 2 .* x̂[:, 1] - 2 .* x̂[:, 2])),
# :( x̂[:, 1] .* x̂[:, 2] .* ( -3. + 2 .* x̂[:, 1] + 2 .* x̂[:, 2])),
# :( tmp1    .* x̂[:, 2] .* (  1  + 2 .* x̂[:, 1] - 2 .* x̂[:, 2])),
# :( 4       .* tmp1 .* tmp2    .* x̂[:, 1]),
# :(-4       .* tmp2 .* x̂[:, 1] .* x̂[:, 2]),
# :(-4       .* tmp1 .* x̂[:, 1] .* x̂[:, 2]),
# :( 4       .* tmp1 .* tmp2    .* x̂[:, 2]),
let lsfns = [
    :(px0 .* py0),
    :(px2 .* py0),
    :(px2 .* py2),
    :(px0 .* py2),
    :(px1 .* py0),
    :(px2 .* py1),
    :(px1 .* py2),
    :(px0 .* py1),
    :(px1 .* py1),
  ],
  grad_lsfns = [
    (:(dpx0 .* py0), :(px0 .* dpy0)),
    (:(dpx2 .* py0), :(px2 .* dpy0)),
    (:(dpx2 .* py2), :(px2 .* dpy2)),
    (:(dpx0 .* py2), :(px0 .* dpy2)),
    (:(dpx1 .* py0), :(px1 .* dpy0)),
    (:(dpx2 .* py1), :(px2 .* dpy1)),
    (:(dpx1 .* py2), :(px1 .* dpy2)),
    (:(dpx0 .* py1), :(px0 .* dpy1)),
    (:(dpx1 .* py1), :(px1 .* dpy1))
  ]
  for (i, (fn, grad)) in enumerate(zip(lsfns, grad_lsfns))
    @eval begin
        @generate_sisd @inline @pure function local_shape_function(
            ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"4-node quadrangle", $(i)},
             x̂::SMatrix{@VectorWidth(num_points), 2, T})::SVector{num_points, T} where {T, num_points}
          px0 =  1 .- 3 .* x̂[:, 1] + 2 .* x̂[:, 1] .* x̂[:, 1]
          px1 = -4 .* x̂[:, 1] .* (-1 .+ x̂[:, 1])
          px2 = -x̂[:, 1] .+ 2 .* x̂[:, 1] .* x̂[:, 1]
          py0 =  1 .- 3 .* x̂[:, 2] + 2 .* x̂[:, 2] .* x̂[:, 2]
          py1 = -4 .* x̂[:, 2] .* (-1 .+ x̂[:, 2])
          py2 = -x̂[:, 2] .+ 2 .* x̂[:, 2] .* x̂[:, 2]
          $(fn)
        end
        @generate_sisd @inline @pure function grad_local_shape_function(
            ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"4-node quadrangle", $(i)},
             x̂::SMatrix{@VectorWidth(num_points), 2, T})::SMatrix{num_points, 2, T} where {T, num_points}
          px0 = 1 .- 3 .* x̂[:, 1] + 2 .* x̂[:, 1] .* x̂[:, 1]
          px1 = -4 .* x̂[:, 1] .* (-1 .+ x̂[:, 1])
          px2 = -x̂[:, 1] .+ 2 .* x̂[:, 1] .* x̂[:, 1]
          py0 = 1 .- 3 .* x̂[:, 2] + 2 .* x̂[:, 2] .* x̂[:, 2]
          py1 = -4 .* x̂[:, 2] .* (-1 .+ x̂[:, 2])
          py2 = -x̂[:, 2] .+ 2 .* x̂[:, 2] .* x̂[:, 2]
          dpx0 = -3 + 4 .* x̂[:, 1]
          dpx1 =  4 - 8 .* x̂[:, 1]
          dpx2 = -1 + 4 .* x̂[:, 1]
          dpy0 = -3 + 4 .* x̂[:, 2]
          dpy1 =  4 - 8 .* x̂[:, 2]
          dpy2 = -1 + 4 .* x̂[:, 2]
          hcat($(grad[1]), $(grad[2]))
        end
      end
    end
end
