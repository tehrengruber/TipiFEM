using ComputedFieldTypes
using MemberFunctions
using TraitDispatch

using LinearAlgebra
using TipiFEM.Meshes: midpoint
using TipiFEM.Utils: tparam, type_scatter

import Base: length, size, getindex
import TipiFEM.Meshes: cell_type, real_type, number_of_cells, cells

@computed struct LocalFEFunction{B <: FEBasis, K <: Cell, REAL_ <: Real} <: AbstractArray{REAL_, 1}
    data::NTuple{number_of_local_shape_functions(B(), K()), REAL_}

    function LocalFEFunction{B, K, REAL_}(in::NTuple{N, <:Real}) where {N, B <: FEBasis, K <: Cell, REAL_ <: Real}
        new(in)
    end
end

@typeinfo basis_type(::Type{T}) where T <: LocalFEFunction = tparam(T, 1)
@typeinfo cell_type(::Type{T}) where T <: LocalFEFunction = tparam(T, 2)
@typeinfo real_type(::Type{T}) where T <: LocalFEFunction = tparam(T, 3)
@typeinfo coeffs_type(::Type{T}) where T <: LocalFEFunction = SVector{number_of_local_shape_functions(basis(T), cell_type(T)()), real_type(T)}
@typeinfo basis(::Type{T}) where T <: LocalFEFunction = basis_type(T)()

# array interface (incomplete)
@typeinfo number_of_coeffs(::Type{T}) where T <: LocalFEFunction = length(coeffs_type(T))
@typeinfo length(::Type{T}) where T <: LocalFEFunction = number_of_coeffs(T)
@typeinfo size(::Type{T}) where T <: LocalFEFunction = (number_of_coeffs(T),)
getindex(lfef::LocalFEFunction, i::Int) = lfef.data[i]

"return coefficients as a SVector"
coeffs(lfef::LocalFEFunction) = coeffs_type(lfef)(lfef.data)

function (lfef::LocalFEFunction{B, K})(geo::Geometry{K}, x::SVector{dim, T}) where {B <: FEBasis, K <: Cell, dim, T}
    lfef(K(), global_to_local(geo, x))
end

function (lfef::LocalFEFunction{B, K})(x̂::SVector{dim, T}) where {B, K <: Cell, dim, T}
    dot(coeffs(lfef), local_shape_functions(B(), K(), x̂))
end

# specializations for 1D
function (lfef::LocalFEFunction{B, K})(x̂::T) where {B, K <: Cell, T <: Real}
  @assert dim(K) == 1
  lfef(SVector{1, T}(x̂))
end

function (lfef::LocalFEFunction{B, K})(geo::Geometry{K}, x̂::T) where {B, K <: Cell, T <: Real}
  @assert dim(K) == 1
  lfef(geo, SVector{1, T}(x̂))
end

#################################################################################
#
# DiscontinuousFEFunction
#
@pure function _dcfef_local_function_type(::Type{FESPACE_}, ::Type{T}) where {FESPACE_ <: BrokenFESpace, T <: Real}
  K = active_cells_type(FESPACE_)
  @assert TipiFEM.Meshes.is_concrete_cell_type(K) "Hybrid meshes not supported yet"

  LocalFEFunction{basis_type(FESPACE_), K, T}
end

@pure function _dcfef_local_functions_type(::Type{FESPACE_}, ::Type{T}) where {FESPACE_ <: BrokenFESpace, T <: Real}
  lfn_t = _dcfef_local_function_type(FESPACE_, T)
  mf_t = first(Base.return_types(_dcfef_construct_local_functions, (FESPACE_, Type{lfn_t}, Array{T, 2})))
  @assert isconcretetype(mf_t)
  mf_t
end

function _dcfef_construct_local_functions(fespace, lfn_t, coeffs)
  MeshFunction(active_cells(fespace), reshape(reinterpret(fulltype(lfn_t), coeffs), (size(coeffs, 2),)))
end

@computed struct DiscontinuousFEFunction{FESPACE_ <: BrokenFESpace, T <: Real}
  fespace::FESPACE_
  coeffs::Array{T, 2}
  local_functions::_dcfef_local_functions_type(FESPACE_, T)

  function DiscontinuousFEFunction{FESPACE_, T}(fespace::FESPACE_) where {FESPACE_ <: BrokenFESpace, T <: Real}
    lfn_t = _dcfef_local_function_type(FESPACE_, T)
    coeffs = Array{T, 2}(undef, (number_of_coeffs(lfn_t), length(active_cells(fespace))))
    DiscontinuousFEFunction{FESPACE_, T}(fespace, coeffs)
  end

  function DiscontinuousFEFunction{FESPACE_, T}(fespace::FESPACE_, coeffs::Array{T, 2}) where {FESPACE_ <: BrokenFESpace, T <: Real}
    let K = active_cells_type(FESPACE_),
        lfn_t = _dcfef_local_function_type(FESPACE_, T)
      @assert TipiFEM.Meshes.is_concrete_cell_type(K) "Hybrid meshes not supported yet"
      @assert(length(active_cells(fespace)) == size(coeffs, 2),
        "The number of columns of the coefficient matrix must equal the number of active cells in the finite element space")
      local_functions = _dcfef_construct_local_functions(fespace, lfn_t, coeffs)
      new{FESPACE_, T}(fespace, coeffs, local_functions)
    end
  end
end

function DiscontinuousFEFunction(fespace::FESPACE_, args...) where FESPACE_ <: BrokenFESpace
  DiscontinuousFEFunction{FESPACE_, Float64}(fespace, args...)
end

@member local_functions(fefun::DiscontinuousFEFunction) = local_functions
@member cells(fefun::DiscontinuousFEFunction) = active_cells(fespace)
@member number_of_cells(fefun::DiscontinuousFEFunction) = length(active_cells(fespace))
@member mesh(fefun::DiscontinuousFEFunction) = mesh(fespace)
@member fespace(fefun::DiscontinuousFEFunction) = fespace
@member coeffs(fefun::DiscontinuousFEFunction) = fefun.coeffs

function getindex(fefun::DiscontinuousFEFunction, id::Id)
  local_functions(fefun)[id]
end

function project!(f::Function, fefn::DiscontinuousFEFunction)
  @assert tparam(basis_type(fefn.fespace), :basis_type) == :Legendre
  let msh = mesh(fefn), lfs = local_functions(fefn)
    for (id, geo) in graph(geometry(msh, cells(fefn)))
      lfs[id, 1] = f(midpoint(geo))
      lfs[id, 2] = f(local_to_global(geo, 1.)[1])-f(local_to_global(geo, 0.)[1])
    end
  end
end

function sample(fefun::DiscontinuousFEFunction, x̂s=0:0.1:1)
  xs = Vector{real_type(mesh(fefun))}(undef, length(x̂s)*number_of_cells(fefun))
  ys = Vector{real_type(mesh(fefun))}(undef, length(x̂s)*number_of_cells(fefun))
  for (i, (geo, lfef)) in enumerate(zip(geometry(mesh(fefun), cells(fefun)), local_functions(fefun)))
    offset = (i-1)*length(x̂s)
    for (j, x̂) in enumerate(x̂s)
      xs[offset+j] = local_to_global(geo, x̂)[1]
      ys[offset+j] = lfef(x̂)
    end
  end
  (xs, ys)
end

for op in [:+, :-, :*, :/]
  @eval begin
    import Base: $(op)
    function $(op)(f1::F, f2::F) where F <: DiscontinuousFEFunction
      @assert fespace(f1) == fespace(f2) "merging of finite element spaces not supported yet"
      F(fespace(f1), $(op).(f1.coeffs, f2.coeffs))
    end
    function $(op)(s::T, f::F) where {T <: Real, F <: DiscontinuousFEFunction}
      F(fespace(f), $(op).(s, f.coeffs))
    end
    function $(op)(f::F, s::T) where {T <: Real, F <: DiscontinuousFEFunction}
      F(fespace(f), $(op).(f.coeffs, s))
    end
  end
end

function l2_norm(fefun::DiscontinuousFEFunction)
  # todo: ensure that transformation from K to K̂ is affine as otherwise
  #  this way of computing the L2 norm is wrong

  # compute mass matrix
  M = mass_matrix(basis(fespace(fefun)), active_cells_type(fespace(fefun))())

  result = 0.
  for (lfef, geo) in zip(local_functions(fefun), geometry(mesh(fefun), cells(fefun)))
    u = coeffs(lfef)
    result += volume(geo)*u'*M*u
  end

  return result
end
