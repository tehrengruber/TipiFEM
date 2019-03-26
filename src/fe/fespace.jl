using TipiFEM.Meshes: HeterogenousMeshFunction, dim_t
using TipiFEM.Utils: IndexMapping, tparam
using Base: return_types

"""
Finite Element Space

 - B: basis specification
 - M: Mesh type
 - D: DofHandler

For cell_type(M) beeing Polytope"3-node triangle" this is the following function space:
```math
  { v ∈ C^k(X) : v_{|C} \\in {\\cal P}(C) \\forall C \\in {\\cal M} }
```
"""
@computed struct FESpace{B <: FEBasis, M <: Mesh, D <: DofHandler, ID_ITER <: IdIterator}
  # specification of the basis functions
  basis::B

  # mesh instance (stores all cell properties of the dofs)
  mesh::M

  # degree of freedom handler / local to global map
  dofh::D

  # fixed interpolation nodes / inactive degrees of freedom and their value
  constraints::Vector{IndexMapping} # todo: use Vector{<: IndexMapping{Int}}

  # todo: to construct the active cells use the parent mesh
  # todo: whether the active_cells_mask mesh function is simple depends on the parent mesh
  #        therefore we need to store the type of the parent mesh in the mesh

  active_cells::ID_ITER

  # inactive interpolation nodes / inactive degrees of freedom
  active_cells_mask::first(return_types(MeshFunction,
    (Type{Union{skeleton(element_type(M))...}}, Type{Bool})))
end

#
# Constructors
#
"construct a finite element space"
function FESpace(basis::B, mesh::M) where {B <: FEBasis, M <: Mesh}
  # construct active cell mask
  active_cells_mask = MeshFunction(Union{skeleton(element_type(M))...}, Bool)
  # construct dof handler
  d = DofHandler(mesh, basis)
  # all cells of the mesh are active
  active_cells = elements(mesh)
  # construct fespace
  fespace = FESpace{B, M, typeof(d), typeof(active_cells)}(basis, mesh, d,
    Vector{IndexMapping}(), active_cells, active_cells_mask)
  # mark all cells active
  Ks = element_type(mesh)
  for Cs in skeleton(Ks)
    for C in uniontypes(Cs)
      let cells = cells(mesh, C())
        set_domain!(active_cells_mask[C], cells)
        set_image!(active_cells_mask[C], fill!(Vector{Bool}(length(cells)), true))
      end
    end
  end
  fespace
end

"construct a finite element space"
function FESpace(basis::B, mesh::M, active_cells::ID_ITER) where {B <: FEBasis, M <: Mesh, ID_ITER <: IdIterator}
  active_cells_mask = MeshFunction(Union{skeleton(element_type(M))...}, Bool)
  # construct dof handler
  d = DofHandler(mesh, basis)
  # construct fespace
  fespace = FESpace{B, M, typeof(d), ID_ITER}(basis, mesh, DofHandler(mesh, basis),
    Vector{IndexMapping}(), active_cells, active_cells_mask)
  # first mark all mesh cells inactive
  Ks = element_type(mesh)
  for Cs in skeleton(Ks)
    for C in uniontypes(Cs)
      let cells = cells(mesh, C())
        set_domain!(active_cells_mask[C], cells)
        set_image!(active_cells_mask[C], fill!(Vector{Bool}(length(cells)), false))
      end
    end
  end
  # then mark all active cells active
  mark_active!(fespace, active_cells)
  fespace
end

#
# field accessors
#

"get mesh field of the finite element space `fespace`"
mesh(fespace::FESpace) = fespace.mesh

"get constraints of the finite element space"
constraints(fespace::FESpace) = fespace.constraints

"get active cells mask of the finite element space"
active_cells_mask(fespace::FESpace) = fespace.active_cells_mask

"get active cells of the finite element space"
active_cells(fespace::FESpace) = fespace.active_cells

"get basis of the finite element space"
basis(fespace::FESpace) = fespace.basis

"get basis of the finite element space"
basis(::Type{T}) where T <: FESpace = tparam(T, 1)()

"get dofhandler of the finite element space"
dofh(fespace::FESpace) = fespace.dofh

#import Base.intersect
#
#function intersect(fespace1::FESpace, fespace2::FESpace)
#  assert(basis(fespace1) == basis(fespace2))
#  assert(mesh(fespace1) == mesh(fespace2))
#  result = FESpace(basis(fespace1), mesh(fespace1), active_cells(fespace1))
#  result.constraints = vcat(fespace1.constraints, fespace2.constraints)
#  result.active_cells_mask = fespace1.active_cells_mask .& fespace2.active_cells_mask
#  result
#end

#
# functionality
#
"return the number of degrees of freedom"
function number_of_dofs(fespace::FESpace)
  let mesh = mesh(fespace), basis=basis(fespace), K = element_type(mesh)
    mapreduce(+, flatten(type_scatter(skeleton(K)))) do C
      number_of_cells(mesh, C()) * multiplicity(basis, C())
    end
  end
end

"add constraints"
function add_constraints!(fespace::FESpace, interp_indmap::IndexMapping{InterpolationNodeIndex, T}) where T <: Number
  # mark all constrained cells inactive
  mark_inactive!(fespace, map(interp_node_idx -> interp_node_idx.cid, indices(interp_indmap)))
  # construct a new index mapping from constrained dof indices to their values
  let dofh = dofh(fespace),
    indmap = IndexMapping{Int, T}()
    for (i, v) in zip(indices(interp_indmap), values(interp_indmap))
      push!(indmap, dofh[i], v)
    end
    push!(constraints(fespace), indmap)
    indmap
  end
end

#@generated interpolation_nodes{C <: Cell}(fespace::FESpace, ::Type{C}; result=[]) = quote
#  let mesh = mesh(fespace),
#      dofh = dofh(fespace),
#      basis = basis(fespace),
#      interpolation_nodes = $(interpolation_nodes(basis(fespace), C())),
#      local_indices = map(local_index, interpolation_nodes)
#      local_coordinates = map(coordinates, interpolation_nodes)
#    for (cid, geo) in graph(geometry(mesh, C()))
#      for (lidx, local_coordinate) in zip(local_indices, local_coordinates)
#        interp_node_index = InterpolationNodeIndex(cid, lidx)
#        interp_node = InterpolationNode(interp_node_index, local_to_global(geo, local_coordinate))
#        push!(result, interp_node)
#      end
#    end
#  end
#  result
#end

#@generated function interpolation_nodes(fespace::FESpace)
#  # get all interpolation nodes in the interior of codimension zero cells
#  expr = Expr(:block)
#  for K in uniontypes(element_type(tparam(fespace, 2)))
#    internal_interpolation_nodes = map(interpolation_node, internal_dofs(basis(fespace), K()))
#    push!(expr.args, quote
#      local_indices = $(map(local_index, internal_interpolation_nodes))
#      local_coordinates = $(map(coordinates, internal_interpolation_nodes))
#      for (cid, geo) in graph(geometry(mesh(fespace), $(K)))
#        for (lidx, local_coordinate) in zip(local_indices, local_coordinates)
#          interp_node_index = InterpolationNodeIndex(cid, lidx)
#          interp_node = InterpolationNode(interp_node_index, local_to_global(geo, local_coordinate))
#          push!(result, interp_node)
#        end
#      end
#    end)
#  end
#  quote
#    result = []
#    # get all interpolation nodes on the facets of codimension zero cells
#    interpolation_nodes(fespace, facet(element_type(mesh(fespace))), result=result)
#
#    $(expr)
#
#    result
#  end
#end

"get all interpolation nodes on the cells `cells`"
function interpolation_nodes(fespace::FESpace, cells::IdIterator)
  # todo: fix that interpolation nodes appear multiple times
  let mesh = mesh(fespace),
      dofh = dofh(fespace),
      node_index_set = Set(InterpolationNodeIndex[]),
      result = IndexMapping{InterpolationNodeIndex, SVector{world_dim(mesh), real_type(mesh)}}(),
      basis = basis(fespace)
    foreach(decompose(cells)) do cells
      K = cell_type(cells)
      for (cid, geo) in graph(geometry(mesh, cells))
        for interp_node in interpolation_nodes(basis, K())
          #push!(result, dofs[index(interp_node)], local_to_global(geo, coordinates(interp_node))
          let index = index(cid, interp_node)
            # todo: using a set is quite slow
            if !(index ∈ node_index_set)
              push!(node_index_set, index)
              push!(result, index, local_to_global(geo, coordinates(interp_node)))
            end
          end
        end
      end
    end
    result
  end
end

#"get all interpolation nodes of the finite element space"
interpolation_nodes(fespace::FESpace) = interpolation_nodes(fespace, fespace.active_cells)

"""
Return a vector of booleans where the `j`-th element is true if the `j`-th
local interpolation node is active
"""
@generated function active_dofs(fespace::FESpace{FEBasis{:Lagrangian, order}}, i::Id{K}) where {order, K <: Cell}
  n=number_of_interpolation_nodes(basis(fespace), K())
  #
  # build an expression that processes the internal dofs
  #
  # get indices of all internal dofs
  internal_dof_indices = map(local_index, internal_dofs(basis(fespace), K()))
  # build the expression
  internal_dofs_expr = quote
    internal_dof_indices = $(internal_dof_indices)
    # iterate over the indices of internal dofs
    for dof_index in internal_dof_indices
      active_dofs[dof_index] = active_cells_mask[i]
    end
  end

  #
  # build an expression that processes the internal dofs
  #
  boundary_dofs_expr = Expr(:block)
  # now process all boundary degrees of freedom
  for d in 0:dim(K)-1
    # get all boundary dofs attached to cells of dimension d
    let boundary_dofs = Base.Iterators.filter(
        boundary_dofs(basis(fespace), K())) do dof
        dim(attached_cell(dof)) == d
      end
      # get the indices of the boundary dofs
      boundary_dofs_indices = map(local_index, boundary_dofs)
      # get to which face each boundary dof is attached to
      attached_faces = map(attached_face, boundary_dofs)
      # build the expression
      push!(boundary_dofs_expr.args, quote
        boundary_dofs_indices = $(boundary_dofs_indices)
        attached_faces = $(attached_faces)
        # get connectivity from K to C of the cell i
        conn = connectivity(mesh(fespace), $(dim_t(K)), $(Dim{d}()))[i]
        # iterate over all boundary dofs
        for (dof_index, face) in zip(boundary_dofs_indices, attached_faces)
          # get the cell id that belongs to the degree of freedom
          j = conn[face]
          # save whether this cell is active
          active_dofs[dof_index] = active_cells_mask[j]
        end
      end)
    end
  end
  quote
    let active_cells_mask = active_cells_mask(fespace),
        active_dofs = zeros(MVector{$(n), Bool})
      # process internal dofs
      $(internal_dofs_expr)
      # process boundary dofs
      $(boundary_dofs_expr)

      active_dofs
    end
  end
end
#function active_dofs(fespace::FESpace{FEBasis{:Lagrangian, order}}, i::Id{K}) where {order, K <: Cell}
#  let n=number_of_interpolation_nodes(basis(fespace), K()),
#      active_dofs = zeros(MVector{n, Bool}),
#      active_cells_mask = active_cells_mask(fespace)
#    # first process all internal degrees of freedom
#    # get the local indices of all internal dofs
#    internal_dof_indices = map(local_index, internal_dofs(basis(fespace), K()))
#    for dof_index in internal_dof_indices
#      active_dofs[dof_index] = active_cells_mask[i]
#    end
#    # now process all boundary degrees of freedom
#    for d in 0:dim(K)-1
#      # get all boundary dofs attached to cells of dimension d
#      let boundary_dofs = filter(boundary_dofs(basis(fespace), K())) do dof
#          dim(attached_cell(dof)) == d
#        end
#        # get the indices of the boundary dofs
#        boundary_dofs_indices = map(local_index, boundary_dofs)
#        # get to which face each boundary dof is attached to
#        attached_faces = map(attached_face, boundary_dofs)
#        # get connectivity from K to C of the cell i
#        conn = connectivity(mesh(fespace), dim_t(K), Dim{d}())[i]
#        # iterate over all boundary dofs
#        for (dof_index, face) in zip(boundary_dofs_indices, attached_faces)
#          # get the cell id that belongs to the degree of freedom
#          j = conn[face]
#          # save whether this cell is active
#          active_dofs[dof_index] = active_cells_mask[j]
#        end
#      end
#    end
#    active_dofs
#  end
#end

## then process all degrees of freedom on the boundary of K
#for d in 0:dim(K)-1
#  # get information about the interpolation nodes in the interor of C
#  let interpolation_nodes = filter(interpolation_nodes(basis(fespace), K())) do node
#      dim(attached_cell(node)) == d
#    end
#    # get connectivity from K to C of the cell i
#    conn = connectivity(mesh(fespace), dim_t(K), Dim{d}())[i]
#    # iterate over all interpolation nodes in the interor of C
#    for interpolation_node in interpolation_nodes
#      # get the cell id that belongs to the interpolation node
#      j = conn[local_cell_index(interpolation_node)]
#      active_dofs[index(interpolation_node)] = active_cells_mask[j]
#    end
#  end
#end

mark_active!(fespace::FESpace, ids::IdIterator) = mark!(fespace, ids; state=true)

"Mark all cells with ids in `ids` and their faces either active or inactive"
function mark!(fespace::FESpace, ids::HomogeneousIdIterator{K}; state=false) where K <: Cell
  # first mark all cells with id in `ids` inactive
  let active_cells_mask = active_cells_mask(fespace)[K]
    for id in ids
      active_cells_mask[id] = state
    end
  end
  # now mark all faces of cells with id in `ids` inactive
  for i in 0:dim(K)-1
    # get the connectivity from K to cells with dim i
    conn = connectivity(mesh(fespace), K(), Dim{i}())
    for C in uniontypes(subcell(K, Dim{i}()))
      # constrain the active cells to type C
      active_cells_mask_constrained = active_cells_mask(fespace)[C]
      # iterate over all inactive cells
      for id in ids
        # iterate over all cells incident to the inactive cells
        for sid in conn[id]
          # mark the incident cell as inactive
          active_cells_mask_constrained[sid] = state
        end
      end
    end
  end
end

mark!(fespace::FESpace, ids::HeterogenousIdIterator; state=false) = foreach(ids -> mark!(fespace, ids, state=state), decompose(ids))

mark_inactive!(fespace::FESpace, ids::IdIterator) = mark!(fespace, ids, state=false)

function interpolate(f::Function , fespace::FESpace)
  f_int = map(f, geometry(mesh(fespace), dofs))
  for constrain in constraints(fespace)
    f_int[domain(constrain)] = image(constrain)
  end
  # todo remove inactive cell_groups
end

"""
Space of p-th degree Lagrangian finite element functions on M

{ v ∈ C^k(X) : v_{|C} \\in {\\cal P}(C) \\forall C \\in {\\cal M} }
"""
const LagrangianFESpace{p, M} = FESpace{FEBasis{:Lagrangian, p}, M}

#FunctionSpace{Basis{:Lagrangian, 2}, 1, Union{Polytope"3-node", Polytope"3-node quadrangle"}}
