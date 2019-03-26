struct CellRef{CELL_ <: Cell, MESH_ <: Mesh}
  mesh::MESH_
  idx::Id{CELL_}
end

CellRef{CELL_ <: Cell, MESH_ <: Mesh}(mesh::MESH_, index::Id{CELL_}) =
  CellRef{CELL_, MESH_}(mesh, index)

mesh{CELL_ <: Cell, MESH_ <: Mesh}(cell::CellRef{CELL_, MESH_}) = cell.mesh
index(cell::CellRef) = cell.idx

cell_type{K <: Cell, MESH_ <: Mesh}(::CellRef{K, MESH_}) = K
mesh_type{K <: Cell, MESH_ <: Mesh}(::CellRef{K, MESH_}) = MESH_

#bitstype sizeof(Int) CellRef{K <: Cell, MESH_ <: Mesh, msh} <: Integer
