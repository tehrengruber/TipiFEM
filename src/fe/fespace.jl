using TipiFEM.Meshes.HeterogenousMeshFunction

using Base.return_types

"""
Finite Element Space

 - B: basis specification
 - M: Mesh type
 - D: DofHandler

For cell_type(M) beeing Polytope"3-node triangle" this is the following function space:
\[
  { v ∈ C^k(X) : v_{|C} \in {\cal P}(C) \forall C \in {\cal M} }
\]
"""
@computed struct FESpace{B <: FEBasis, M <: Mesh, D <: DofHandler}
  # specification of the basis functions
  basis::B

  # mesh instance (stores all cell properties of the dofs)
  mesh::M

  # degree of freedom handler / local to global map
  dofh::D

  # fixed interpolation nodes / inactive degrees of freedom and their value
  constraints::Vector{Vector{Tuple{Index{vertex(cell_type(M))}, real_type(M)}}}

  # todo: to construct the active cells use the parent mesh
  # todo: whether the active_cells mesh function is simple depends on the parent mesh
  #        therefore we need to store the type of the parent mesh in the mesh

  # inactive interpolation nodes / inactive degrees of freedom
  active_cells::first(return_types(MeshFunction,
    (Type{Union{skeleton(cell_type(M))...}}, Type{Bool},
     Val{issimple(hasparent(M) ? parent_type(M) : M)})))
end

function interpolation_nodes(fespace::FESpace, cells::IdIterator)
  let mesh = mesh(fespace),
      interpolation_nodes = Vector{SVector{world_dim(mesh), calc_type(mesh)}}()
    for (cid, geo) in geometry(cells)
      for interp_node in interpolation_nodes(basis(fespace), cell_type(cid))
        push!(interpolation_nodes, local_to_global(geo, coordinates(interp_node)))
      end
    end
  end
  interpolation_nodes
end

function FESpace{B <: FEBasis, M <: Mesh, D <: DofHandler}(basis::B, mesh::M, dofh::D)
  active_cells = MeshFunction(Union{skeleton(cell_type(M))...}, Bool,
   Val{issimple(hasparent(M) ? parent_type(M) : M)}())
  # construct fespace
  fespace = FESpace{B, M, D}(basis, mesh, dofh,
    fieldtype(fulltype(FESpace{B, M, D}), :constraints)(),
    active_cells)
  # if the mesh of the fespace has a parent (e.g. it represents the boundary of
  #  another mesh or it is a subdomain of the domain of the original mesh represents)
  #  then we first set all cells of the parent mesh inactive and then activate
  #  all cells again that are part of the child mesh
  if hasparent(mesh)
    # first initialize active cells mesh function with the domain of the parent
    #  mesh and set all of its cells inactive

    Ks = cell_type(mesh)
    for Cs in skeleton(Ks)
      for C in uniontypes(Cs)
        let cells = cells(parent(mesh), C())
          set_domain!(active_cells[C], cells)
          set_image!(active_cells[C], fill!(Vector{Bool}(length(cells)), false))
        end
      end
    end

    for Cs in skeleton(Ks)
      for C in uniontypes(Cs)
        let cells = cells(mesh, C())
          foreach(cells) do cid
            active_cells[C][cid] = true
          end
        end
      end
    end
  # if the mesh has no parent we just mark all cells active
  else
    Ks = cell_type(mesh)
    for Cs in skeleton(Ks)
      for C in uniontypes(Cs)
        let cells = cells(mesh, C())
          set_domain!(active_cells[C], cells)
          set_image!(active_cells[C], fill!(Vector{Bool}(length(cells)), true))
        end
      end
    end
  end
  fespace
end

FESpace{B <: FEBasis, M <: Mesh}(basis::B, msh::M) = FESpace(basis, msh, DofHandler(msh, basis))

active_cells(fespace::FESpace) = fespace.active_cells

basis(fespace::FESpace) = fespace.basis

using TipiFEM.Meshes.dim_t

"""
Return a vector of booleans where the `j`-th element is true if the `j`-th
local interpolation node is active
"""
function active_dofs(fespace::FESpace{FEBasis{:Lagrangian, order}}, i::Index{K}) where {order, K <: Cell}
  let n=number_of_interpolation_nodes(basis(fespace), K()),
      active_dofs = zeros(MVector{n, Bool}),
      active_cells = active_cells(fespace)
    # first process all internal degrees of freedom
    # get the local indices of all internal dofs
    internal_dof_indices = map(index, internal_dofs(basis(fespace), K()))
    for dof_index in internal_dof_indices
      active_dofs[dof_index] = active_cells[i]
    end
    # now process all boundary degrees of freedom
    for d in 0:dim(K)-1
      # get all boundary dofs attached to cells of dimension d
      let boundary_dofs = filter(boundary_dofs(basis(fespace), K())) do dof
          dim(attached_cell(dof)) == d
        end
        # get the indices of the boundary dofs
        boundary_dofs_indices = map(index, boundary_dofs)
        # get to which face each boundary dof is attached to
        attached_faces = map(attached_face, boundary_dofs)
        # get connectivity from K to C of the cell i
        conn = connectivity(mesh(fespace), dim_t(K), Dim{d}())[i]
        # iterate over all boundary dofs
        for (dof_index, face) in zip(boundary_dofs_indices, attached_faces)
          # get the cell id that belongs to the degree of freedom
          j = conn[face]
          # save whether this cell is active
          active_dofs[dof_index] = active_cells[j]
        end
      end
    end
    active_dofs
  end
end

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
#      active_dofs[index(interpolation_node)] = active_cells[j]
#    end
#  end
#end

"""
Mark all cells with ids in `ids` and their faces inactive
"""
function mark_inactive!(fespace::FESpace, ids::IdIterator{K}) where K <: Cell
  # first mark all cells with id in `ids` inactive
  let active_cells_constrained = active_cells(fespace)[K]
    for id in ids
      active_cells_constrained[id] = false
    end
  end
  # now mark all faces of cells with id in `ids` inactive
  for i in 0:dim(K)-1
    # get the connectivity from K to cells with dim i
    conn = connectivity(mesh(fespace), K(), Dim{i}())
    for C in uniontypes(subcell(K, Dim{i}()))
      # constrain the active cells to type C
      active_cells_constrained = active_cells(fespace)[C]
      # iterate over all inactive cells
      for id in ids
        # iterate over all cells incident to the inactive cells
        for sid in conn[id]
          # mark the incident cell as inactive
          active_cells_constrained[sid] = false
        end
      end
    end
  end
end

mesh(fespace::FESpace) = fespace.mesh

constraints(fespace::FESpace) = fespace.constraints

function inactive_cell_groups(fespace::FESpace)
  chain(fespace.inactive_cell_groups)
end

#interpolation_nodes(fespace::FESpace) = mesh(fespace) ∩ fixed_interpolation_nodes

function constrain(fespace::FESpace, constrain::MeshFunction)
  push!(constraints(fespace), constrain)
  fespace
end

function interpolate(f::Function , fespace::FESpace)
  f_int = map(f, geometry(mesh(fespace), dofs))
  for constrain in constraints(fespace)
    f_int[domain(constrain)] = image(constrain)
  end
  # todo remove inactive cell_groups
end

"""
Space of p-th degree Lagrangian finite element functions on M

{ v ∈ C^k(X) : v_{|C} \in {\cal P}(C) \forall C \in {\cal M} }
"""
const LagrangianFESpace{p, M} = FESpace{FEBasis{:Lagrangian, p}, M}

#FunctionSpace{Basis{:Lagrangian, 2}, 1, Union{Polytope"3-node", Polytope"3-node quadrangle"}}
