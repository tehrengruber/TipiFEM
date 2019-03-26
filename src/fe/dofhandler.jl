# In general a DofHandler can be implemented having only identities for
#  codim 0 and codim 1 cells (in another words having only the codim 0, 1 -> dim 0
#  entries populated in the topology array). In 2d however we already have identities
#  for codim 0, 1, 2 cells. Therefore we don't save anything, but get a very
#  slick and easy to grasp implementation of the dofhandler.
using Base: uniontypes
using TipiFEM.Utils: flatten, type_scatter
using ComputedFieldTypes

import Base: getindex

"""
Subtypes of this type store all necessary information to map cells associated
with a degree of freedom to unique integer indices.
"""
abstract type AbstractDofHandler end

"""
This DofHandler assigns integer ranges to each cell type. A unique integer
index is then calculated from a cell index by adding the start value of the
corresponding range and the cell index (in integer form).
"""
@computed struct DofHandler{K <: Cell, BT <: FEBasis, M <: Mesh} <: AbstractDofHandler
  msh::M
  basis::BT
  offset::SVector{length(flatten(type_scatter(skeleton(K)))), Int}

  # todo: check invariant K == cell_type(M)
end

function DofHandler(msh::Mesh{K}, basis::BT) where {K <: Cell, BT <: FEBasis}
  # todo: assert initialized mesh
  # precompute offsets
  sizes = map(C -> number_of_cells(msh, C()) * multiplicity(basis, C()), flatten(type_scatter(skeleton(K))))
  offsets = Array{Int, 1}(undef, length(sizes))
  let offset = 0
    for i in 1:length(offsets)
      offsets[i] = offset
      offset += sizes[i]
    end
  end
  # construct
  DofHandler{K, BT, typeof(msh)}(msh, basis, offsets)
end

"""
Returns an integer offset for a given cell type such that the offset plus a
cell index in integer form, associated with a degree of freedom is unique.
"""
@generated function offset(dof_handler::DofHandler{K}, ::C) where {K <: Cell, C <: Cell}
  i = findfirst(isequal(C), flatten(type_scatter(skeleton(K))))
  i::Int # otherwise this method is not type stable...
  i != 0 || error("attempt get offset for a cell type which is not "
        * "contained in the skeleton of any codim zero cell")
  :(dof_handler.offset[$(i)])
end

#"""
#Given a cell associated with a degree of freedom returns a unique integer index
#"""
#function getindex(dofh::DofHandler, el_idx::Id{K}, ::LocalDOFIndex{idx}) where {K <: Cell, idx}
#  C = associated_cell_type(K, dofh.basis, idx)
#  offset(dofh, C)+convert(Int, connectivity(dofh.msh, K, C)[el_idx][idx])
#end

"""
Given a codim zero cell id returns unique integer indices for all
degrees of freedom on the cell
"""
@generated function getindex(dofh::DofHandler, el_idx::Id{K}) where K <: Cell
  # todo: in 3d this does not work if the faces of K with the same dimension
  #  are all of the same type
  @sanitycheck @assert dim(K) < 3 "not implemented"
  # build an expression that assigns indices to all internal dofs
  expr = Expr(:block)
  for C in skeleton(K)[1:end-1]
    push!(expr.args, quote
      let C=$(C), conn = connectivity(dofh.msh, K(), dim_t(C))[el_idx]
        for i in 1:face_count(K, C)
          for j in 0:multiplicity(dofh.basis, C())-1
            @inbounds result[pos] = offset(dofh, C())+convert(Int, conn[i])+j
            #unsafe_store!(convert(Ptr{Int}, Base.pointer_from_objref(result)), didx, pos)
            pos+=1
          end
        end
      end
    end)
  end
  quote
    n = number_of_local_shape_functions(dofh.basis, K())
    result = zeros(MVector{n, Int})
    pos=1
    # process boundary degrees of freedom
    $(expr)
    # process internal degrees of freedom
    for j in 0:multiplicity(dofh.basis, K())-1
      @inbounds result[pos] = offset(dofh, K())+convert(Int, el_idx)+j
    end
    result
  end
end

function getindex(dofh::DofHandler, interp_node_idx::InterpolationNodeIndex{C}) where C <: Cell
  dofh[interp_node_idx.cid][interp_node_idx.lidx]
end

#"cell type associated with a local dof index"
#associated_cell_type{K <: Cell}(dofh::DofHandler{K}, lidx::LocalDOFIndex) = interpret_local_index(dofh, lidx)[1]
#
#"integer in the range 0 to multiplicity(dofh.basis, associated_cell_type(K))-1"
#local_offset{K <: Cell}(dofh::DofHandler{K}, lidx::LocalDOFIndex) = interpret_local_index(dofh, lidx)[2]
#
#"""
#Given a codim zero cell type and a basis compute the cell type associated to
#the local index and its local offset
#"""
#@Base.pure function interpret_local_index{K <: Cell, lidx}(dofh::DofHandler{K}, ::LocalDOF{lidx})
#  lidx_stop = 0
#  local_offset = 0
#  # iterate over the skeleton of K in order of increasing dimension
#  for T in flatten(type_scatter(skeleton(K)))
#    # every face has multiplicity many dofs
#    lidx_stop+=face_count(K, T) * multiplicity(dofh.basis, T())
#    if lidx_stop>=lidx
#      return (T, local_offset)
#    end
#    local_offset = lidx-lidx_stop
#  end
#  error("attempt get associated cell type of an invalid LocalDOFIndex")
#end
