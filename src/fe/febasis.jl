using Base: @pure
using MacroTools
using TraitDispatch
using StaticArrays

using TipiFEM.Utils: canonicalize, @generate_sisd, @typeinfo, tparams, fastdropdims
using TipiFEM.PolytopalMesh
using TipiFEM.Meshes
using TipiFEM.Meshes: Cell

abstract type AbstractFEBasis end

"""
Singleton type specifing a basis of a finite element space.

- `basis_type`: Symbol describing the type of basis (e.g. `:Lagrangian`)
- `approx_order`: Polynomial degree

The actual properties of the basis are defined by functions dispatching
on this type (i.e. multiplicity, local_shape_function).
"""
struct FEBasis{basis_type, approx_order} <: AbstractFEBasis end

"""
Local degree of freedom
"""
struct LocalDOF{B <: FEBasis, K <: Cell, idx} end

basis(::LocalDOF{B}) where B = B()

basis_type(::LocalDOF{B}) where B = B

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

# for a modal basis all dofs are associated with the interior of the cell
#  hence the multiplicity for a vertex is zero ()
multiplicity(::FEBasis{:Legendre, order}, ::Polytope"1-node point") where order = 0
multiplicity(::FEBasis{:Legendre, order}, ::Polytope"2-node line") where order = (order+1)
multiplicity(::FEBasis{:Legendre, order}, ::Polytope"4-node quadrangle") where order = (order+1)*(order+1)

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
          ::Type{T}=Float64) where T <: Real
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
@pure function local_shape_functions(basis::FEBasis, ::K) where K <: Cell
  [local_shape_function(LocalDOF{basis, K, i}()) for i in 1:number_of_local_shape_functions(basis, K())]
end

"""
Evaluate all local shape functions at points `x̂s` (local coordinates)
returning a vector of dimension `num_fns` with eltype
`SVector{num_points, T}`. The i-th element of the outer vector is another
vector containing the values of the i-th shape function at `x̂s`.
"""
@generated function local_shape_functions(basis::FEBasis, ::K,
    x̂s::Union{SMatrix, SVector}) where {K <: Cell}
  #assert(local_dim == dim(K))
  num_fns=number_of_local_shape_functions(basis(), K())
  args = (:(local_shape_function(LocalDOF{typeof(basis), K, $i}(), x̂s)) for i in 1:num_fns)
  :(SVector($(args...)))
end

@generated function grad_local_shape_functions(basis::FEBasis, ::K,
    x̂s::Union{SMatrix, SVector}) where {K <: Cell}
  #assert(local_dim == dim(K))
  num_fns=number_of_local_shape_functions(basis(), K())
  args = (:(grad_local_shape_function(LocalDOF{typeof(basis), K, $i}(), x̂s)) for i in 1:num_fns)
  :(SVector($(args...)))
end

macro define_unvectorized_shape_functions(fn, rules...)
  ex = Expr(:block)
  for rule in rules
    # parse rule
    @capture(rule, ((trait_, conversion_rule_, (targs__,)) -> expr_)) || error("invalid rule format")
    # parse conversion rule
    @capture(conversion_rule, (coord_in_t_ → coord_out_t_)) || error("invalid conversion rule format")
    # generate
    fnex = quote
      @traitfn function local_shape_function(ldof::LocalDOF{B, K}, x̂::$(coord_in_t)) where {B <: FEBasis, K <: Cell, $(targs...); $trait}
        $(Utils.@replace(expr, slot, local_shape_function(ldof, $(coord_out_t)(x̂))))
      end

      @traitfn function grad_local_shape_function(ldof::LocalDOF{B, K}, x̂::$(coord_in_t)) where {B <: FEBasis, K <: Cell, $(targs...); $trait}
        $(Utils.@replace(expr, slot, grad_local_shape_function(ldof, $(coord_out_t)(x̂))))
      end

      @traitfn function local_shape_functions(basis::B, ::K, x̂::$(coord_in_t)) where {B <: FEBasis, K <: Cell, $(targs...); $trait}
        map(N -> $(Utils.@replace(expr, slot, N)), local_shape_functions(basis, K(), $(coord_out_t)(x̂)))
      end

      @traitfn function grad_local_shape_functions(basis::B, ::K, x̂::$(coord_in_t)) where {B <: FEBasis, K <: Cell, $(targs...); $trait}
        map(∇N -> $(Utils.@replace(expr, slot, ∇N)), grad_local_shape_functions(basis, K(), $(coord_out_t)(x̂)))
      end
    end
    push!(ex.args, fnex)
  end
  esc(ex)
end

@define_unvectorized_shape_functions(local_shape_function,
    ((dim(K)==1 && is_scalar_basis(B)), T → SVector{1, T}, (T <: Real,)) -> fastdropdims(slot, (1,)),
    ((dim(K)==1 && !is_scalar_basis(B)), T → SMatrix{1, 1, T}, (T <: Real,)) -> fastdropdims(slot, (1,)),
    (dim(K)!=1, (SVector{local_dim, T} → SMatrix{1, local_dim, T}), (T <: Real, local_dim,)) -> fastdropdims(slot, (1,)))

#
# Lagrangian FE
#
#
# p = 1
#
@typeinfo is_scalar_basis(::Type{<:FEBasis{:Lagrangian}}) = true

#@define_basis(FEBasis{:Lagrangian, 1}, Polytope"2-node line", (1-x̂, x̂))
#@define_basis(FEBasis{:Lagrangian, 2}, Polytope"2-node line", (2(x̂-1)(x̂-0.5), 2x̂(x̂-0.5), 4(1-x̂)x̂))

let lsfs = [:(1 .- x̂), :(x̂)], grad_lsfs = [-1, 1]
  # given the local shape functions as expressions iterate over them and generate
  #  the corresponding function definitions
  for (i, (lsf, grad_lsf)) in enumerate(zip(lsfs, grad_lsfs))
    @eval @traitfn @inline @pure function local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"2-node line", $(i)},
        x̂::SVector{num_points, T}) where {T, num_points}
      $(lsf)
    end
    @eval @traitfn @inline @pure function grad_local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"2-node line", $(i)},
         x̂::SVector{num_points, T}) where {T, num_points}
      $(grad_lsf)*ones(SVector{num_points, T})
    end
  end
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
    @eval @inline @pure function local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"3-node triangle", $(i)},
        x̂::SMatrix{num_points, 2, T})::SVector{num_points, T} where {T, num_points}
      $(lsf)
    end
    @eval @inline @pure function grad_local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"3-node triangle", $(i)},
         x̂::SMatrix{num_points, 2, T})::SMatrix{num_points, 2, T} where {T, num_points}
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
    @eval @inline @pure function local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"4-node quadrangle", $(i)},
        x̂::SMatrix{num_points, 2, T})::SVector{num_points, T} where {T, num_points}
      $(fn)
    end
    @eval @inline @pure function grad_local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 1}, Polytope"4-node quadrangle", $(i)},
         x̂::SMatrix{num_points, 2, T})::SMatrix{num_points, 2, T} where {T, num_points}
      # todo: use repeat iteration instead of copying
      hcat($(grad[1]), $(grad[2]))
    end
  end
end

#
# p=2
#
# todo: use devectorize
# edges
let lsfs = [:(2*(x̂.-1).*(x̂.-0.5)), :(2*x̂.*(x̂ .- 0.5)), :(4*(1 .- x̂).*x̂)],
    grad_lsfs = [:(4x̂-3), :(4x̂-1), :(4-8x̂)]
  # given the local shape functions as expressions iterate over them and generate
  #  the corresponding function definitions
  for (i, (lsf, grad_lsf)) in enumerate(zip(lsfs, grad_lsfs))
    @eval @traitfn @inline @pure function local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"2-node line", $(i)},
        x̂::SVector{num_points, T}) where {T, num_points}
      $(lsf)
    end
    @eval @traitfn @inline @pure function grad_local_shape_function(
        ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"2-node line", $(i)},
         x̂::SVector{num_points, T}) where {T, num_points}
      $(grad_lsf)*ones(SVector{num_points, T})
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
      (:(zeros(SVector{num_points, T})),   :( 4x̂[:, 2]-1)),
      (:(4 .- 8x̂[:, 1] .- 4x̂[:, 2]),       :(-4x̂[:, 1])),
      (:(4x̂[:, 2]),                        :( 4x̂[:, 1])),
      (:(-4x̂[:, 2]),                       :( 4 .- 4x̂[:, 1] .- 8x̂[:, 2]))
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
        @eval @inline @pure function local_shape_function(
            ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"3-node triangle", $(i)},
             x̂::SMatrix{num_points, 2, T})::SVector{num_points, T} where {T, num_points}
          $(fn)
        end
        @eval @inline @inline @pure function grad_local_shape_function(
            ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"3-node triangle", $(i)},
             x̂::SMatrix{num_points, 2, T})::SMatrix{num_points, 2, T} where {T, num_points}
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
        @inline @pure function local_shape_function(
            ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"4-node quadrangle", $(i)},
             x̂::SMatrix{num_points, 2, T})::SVector{num_points, T} where {T, num_points}
          px0 =  1 .- 3 .* x̂[:, 1] + 2 .* x̂[:, 1] .* x̂[:, 1]
          px1 = -4 .* x̂[:, 1] .* (-1 .+ x̂[:, 1])
          px2 = -x̂[:, 1] .+ 2 .* x̂[:, 1] .* x̂[:, 1]
          py0 =  1 .- 3 .* x̂[:, 2] + 2 .* x̂[:, 2] .* x̂[:, 2]
          py1 = -4 .* x̂[:, 2] .* (-1 .+ x̂[:, 2])
          py2 = -x̂[:, 2] .+ 2 .* x̂[:, 2] .* x̂[:, 2]
          $(fn)
        end
        @inline @pure function grad_local_shape_function(
            ::LocalDOF{FEBasis{:Lagrangian, 2}, Polytope"4-node quadrangle", $(i)},
             x̂::SMatrix{num_points, 2, T})::SMatrix{num_points, 2, T} where {T, num_points}
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

#
# Legendre FE
#
@typeinfo is_scalar_basis(::Type{<:FEBasis{:Legendre}}) = true

# todo: rename to number_of_dofs?
function number_of_local_shape_functions(basis::FEBasis{:Legendre, order}, ::Polytope"2-node line") where order
  order+1
end

let lsfs = [:(ones(SVector{num_points, T}))], grad_lsfs = [0.]
  for (i, (lsf, grad_lsf)) in enumerate(zip(lsfs, grad_lsfs))
    @eval @traitfn @pure function local_shape_function(
        ::LocalDOF{FEBasis{:Legendre, 0}, Polytope"2-node line", $(i)},
        x̂::SVector{num_points, T}) where {T, num_points}
      $(lsf)
    end
    @eval @traitfn @pure function grad_local_shape_function(
        ::LocalDOF{FEBasis{:Legendre, 0}, Polytope"2-node line", $(i)},
         x̂::SVector{num_points, T}) where {T, num_points}
      $(grad_lsf)*ones(SVector{num_points, T})
    end
  end
end

let lsfs = [:(ones(SVector{num_points, T})), :((x̂.-0.5))], grad_lsfs = [0., 1.]
  # given the local shape functions as expressions iterate over them and generate
  #  the corresponding function definitions
  for (i, (lsf, grad_lsf)) in enumerate(zip(lsfs, grad_lsfs))
    @eval @traitfn @pure function local_shape_function(
        ::LocalDOF{FEBasis{:Legendre, 1}, Polytope"2-node line", $(i)},
        x̂::SVector{num_points, T}) where {T, num_points}
      $(lsf)
    end
    @eval @traitfn @pure function grad_local_shape_function(
        ::LocalDOF{FEBasis{:Legendre, 1}, Polytope"2-node line", $(i)},
         x̂::SVector{num_points, T}) where {T, num_points}
      $(grad_lsf)*ones(SVector{num_points, T})
    end
  end
end

# todo: cleanup
mass_matrix(::FEBasis{:Legendre, order}, ::Polytope"2-node line", T::Type=Float64) where order = SDiagonal{2, T}([1, 1/3-1/2+0.25])

#
# Vector basis
#
@typeinfo is_scalar_basis(::Type{<:FEBasis{:Vector}}) = false

@pure underlying_bases(::Type{<:FEBasis{:Vector, Bs}}) where Bs <: Tuple{Vararg{<:FEBasis}} = tparams(Bs)

@pure underlying_bases(::PB) where PB <: FEBasis{:Vector} = map(B -> B(), underlying_bases(PB))

@typeinfo num_components(vector_basis::Type{<:FEBasis{:Vector}}) = length(underlying_bases(vector_basis))

@pure function number_of_local_shape_functions(basis::FEBasis{:Vector}, ::K) where K <: Cell
  mapreduce(+, underlying_bases(basis)) do basis
    number_of_local_shape_functions(basis, K())
  end
end

@generated function associated_dofs(::VecB, ::K) where {VecB <: FEBasis{:Vector}, K <: Cell}
  tuple_ex = Expr(:tuple)
  for B in underlying_bases(VecB)
    for i in 1:number_of_local_shape_functions(B(), K())
      push!(tuple_ex.args, :(LocalDOF{$(B), K, $(i)}()))
    end
  end
  tuple_ex
end

function associated_dof(dof::LocalDOF{<:FEBasis{:Vector}, K, i}) where {i, K <: Cell}
  associated_dofs(basis(dof), K())[i]
end

@generated function associated_components(::VecB, ::K) where {K <: Cell, VecB <: FEBasis{:Vector}}
  tuple_ex = Expr(:tuple)
  for (i, B) in enumerate(underlying_bases(VecB))
    for _ in 1:number_of_local_shape_functions(B(), K())
      push!(tuple_ex.args, i)
    end
  end
  tuple_ex
end

function associated_component(dof::LocalDOF{<:FEBasis{:Vector}, K, i}) where {i, K <: Cell}
  associated_components(basis(dof), K())[i]
end

import Base: vcat
vcat(::B1, ::B2) where {B1 <: FEBasis, B2 <: FEBasis} = FEBasis{:Vector, Tuple{B1, B2}}()

vcat(::B1, ::B2) where {B1 <: FEBasis{:Vector}, B2 <: FEBasis} = FEBasis{:Vector, Tuple{underlying_bases(B1)..., B2}}()
vcat(::B1, ::B2) where {B1 <: FEBasis, B2 <: FEBasis{:Vector}} = FEBasis{:Vector, Tuple{B1, underlying_bases(B2)...}}()

@pure function transformation_vector(dof::LocalDOF{<: FEBasis{:Vector}, K, i}, T::Type) where {K <: Cell, i}
  SVector{num_components(basis(dof)), T}((
    (0 for i in 1:associated_component(dof)-1)...,
    1,
    (0 for i in associated_component(dof)+1:num_components(basis(dof)))...))
end

using Einsum

function local_shape_function(
    dof::LocalDOF{<: FEBasis{:Vector}},
    x̂s::SMatrix{num_points, local_dim, T}) where {num_points, local_dim, T}
  # compute transformation vector
  #  e.g. for a dof associated with the 1. component of the vector valued
  #  shape function with 3 components this is just (1, 0, 0)
  t⃗ = transformation_vector(dof, T)
  # evaluate local shape function of the associated dof
  N = local_shape_function(associated_dof(dof), x̂s)
  # allocate result vector
  N⃗ = zeros(MMatrix{num_points, num_components(basis(dof)), T})
  # compute result
  @einsum N⃗[i, j] = t⃗[j]*N[i]
  # convert into statically sized matrix
  SMatrix{num_points, num_components(basis(dof)), T}(N⃗)
end

# jacobian
function grad_local_shape_function(
    dof::LocalDOF{<: FEBasis{:Vector}, K},
    x̂s::SMatrix{num_points, local_dim, T}) where {K <: Cell, num_points, local_dim, T}
  # compute transformation vector
  #  e.g. for a dof associated with the 1. component of the vector valued
  #  shape function with 3 components this is just (1, 0, 0)
  t⃗ = transformation_vector(dof, T)
  # evaluate local shape function of the associated dof
  ∇N = grad_local_shape_function(associated_dof(dof), x̂s)
  # allocate result vector
  ∇N⃗ = zeros(MArray{Tuple{num_points, num_components(basis(dof)), local_dim}, T})
  # compute result
  @einsum ∇N⃗[i, j, k] = t⃗[j]*∇N[i, k]
  # convert into statically sized matrix
  SArray{Tuple{num_points, num_components(basis(dof)), local_dim}, T}(∇N⃗)
end
