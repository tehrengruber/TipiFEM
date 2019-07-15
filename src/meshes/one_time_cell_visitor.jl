@computed mutable struct OneTimeCellVisitorState{M <: Mesh}
  visitor_id::Int
  mf::_one_time_visitor_mf_type(element_type(M))
end

function _one_time_visitor_mf_type(@nospecialize(K); warn=true)
  warn && @warn "type-unstable version of _one_time_visitor_mf_type called for cell type $(K)"
  first(return_types(MeshFunction,
    (Type{Union{skeleton(K)...}}, Type{Int})))
end

add_cell_initializer() do mod, K
  dim(K) != 0 || return
  @eval mod Base.@pure _one_time_visitor_mf_type(::Type{$(K)}) where K <: Cell = $(_one_time_visitor_mf_type(K, warn=false))
end

function OneTimeCellVisitorState(mesh::M) where M <: Mesh
  mf = MeshFunction(Union{skeleton(element_type(M))...}, Int)

  Ks = element_type(mesh)
  for Cs in skeleton(Ks)
    for C in uniontypes(Cs)
      let cells = cells(mesh, C())
        set_domain!(mf[C], cells)
        set_image!(mf[C], fill!(Vector{Bool}(undef, length(cells)), 0))
      end
    end
  end

  OneTimeCellVisitorState{M}(0, mf)
end

"""
    one_time_cell_visitor(msh::Mesh)

```
ocv = one_time_cell_visitor(msh::Mesh)
for el_id in cells
  for cid in subcells(msh, el_id)
    if ocv.mf[cid]==ocv.visitor_id
      println("cell \$(cid) has already been visited")
    else
      println("cell \$(cid) is visited for the first time")
    end
  end
end
```
"""
function one_time_cell_visitor(msh::M) where M <: Mesh
  ucv = TipiFEM.Meshes.cache_entry(mesh, :one_time_cell_visitor)
  ucv::fulltype(OneTimeCellVisitorState{M})
  ucv.visitor_id += 1

  (ucv.visitor_id, ucv.mf)
end

# todo: add unique(mesh, cells) function

add_cache_initializer(mesh -> OneTimeCellVisitorState(mesh), :one_time_cell_visitor)
