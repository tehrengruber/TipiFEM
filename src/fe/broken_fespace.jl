
# note: for the DG method we don't need a DofHandler (and consequently also no
# active_cells_mask)

"""
The Finite Element space is said to be broken if the dofs associated
with the boundary of a cell do not need to coincide with the corresponding
dofs on the neighbouring cell.
"""
@computed struct BrokenFESpace{B <: FEBasis, M <: Mesh, ID_ITER <: IdIterator}
  # specification of the basis functions
  basis::B

  # mesh instance (stores all cell properties of the dofs)
  mesh::M

  # fixed interpolation nodes / inactive degrees of freedom and their value
  constraints::Dict{Symbol, IndexMapping}

  active_cells::ID_ITER
end

function BrokenFESpace(basis::B, mesh::M, active_cells::I) where {B <: FEBasis, M <: Mesh, I <: IdIterator}
  # construct fespace
  BrokenFESpace{B, M, I}(basis, mesh, Dict{Symbol, IndexMapping}(), active_cells)
end

BrokenFESpace(basis::FEBasis, mesh::Mesh) = BrokenFESpace(basis, mesh, elements(mesh))

"get mesh field of the finite element space `fespace`"
mesh(fespace::BrokenFESpace) = fespace.mesh

"get constraints of the finite element space"
constraints(fespace::BrokenFESpace) = fespace.constraints

"get active cells of the finite element space"
active_cells(fespace::BrokenFESpace) = fespace.active_cells

"get datatype of the basis of the finite element space"
@typeinfo basis_type(::Type{T}) where T <: BrokenFESpace = tparam(T, 1)

"get basis of the finite element space"
basis(::Type{T}) where T <: BrokenFESpace = basis_type(T)()

"get basis of the finite element space"
basis(::T) where T <: BrokenFESpace = basis(T)

id_iterator_type(::Type{T}) where T <: BrokenFESpace = tparam(T, :ID_ITER)

"cell type of the active cells of the finite element space"
@typeinfo active_cells_type(::Type{T}) where T <: BrokenFESpace = cell_type(eltype(tparam(T, 3)))

"mesh type of finite element space"
mesh_type(::Type{T}) where T <: BrokenFESpace = tparam(T, 2)

function number_of_dofs(fespace::BrokenFESpace)
  let mesh = mesh(fespace), basis=basis(fespace), K = element_type(mesh)
    mapreduce(+, type_scatter(K)) do C
      number_of_cells(mesh, C()) * number_of_local_shape_functions(basis, C())
    end
  end
end

function number_of_active_dofs(fespace::BrokenFESpace)
  let mesh = mesh(fespace), basis=basis(fespace), K = element_type(mesh)
    mapreduce(+, type_scatter(K)) do C
      length(active_cells(fespace)) * number_of_local_shape_functions(basis, C())
    end
  end
end
