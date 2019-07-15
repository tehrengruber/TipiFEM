using TipiFEM.Utils: MethodNotImplemented
using Base: @pure

export @Polytope_str, Polytope, vertex_count, @Id_str, @Connectivity_str, vertex, Connectivity

################################################################################
# data-types
################################################################################
abstract type Cell end

################################################################################
# type-traits
################################################################################
# see https://en.wikipedia.org/wiki/N-skeleton
@pure skeleton(::Type{T}) where {T <: Cell} = dim(T) > 0 ? (skeleton(facet(T))..., T) : (T,)
@dim_dispatch @Base.pure skeleton(::Type{T}, ::Dim{d}) where {T <: Cell, d} = skeleton(T)[1:d+1]
@Base.pure function subcell(::Type{T}, ::Codim{cd}) where {T <: Cell, cd}
  if cd == 0 # end recursion
    T
  elseif typeof(T) == Union
    Union{subcell(T.a, Codim{cd}()), subcell(T.b, Codim{cd}())}
  else
    subcell(facet(T), Codim{cd-1}())
  end
end
@Base.pure subcell(::Type{T}, d::Union{Int, Dim}) where {T <: Cell} = subcell(T, Codim{dim(T)-convert(Int, d)}())
@Base.pure subcell(::Type{T}) where {T <: Cell} = subcell(T, Codim{1}())
@Base.pure vertex(::Type{T}) where {T <: Cell} = first(skeleton(T))
@Base.pure vertex_count(::Type{T}) where {T <: Cell} = face_count(T, vertex(T))
@Base.pure facet(::Type{UT}) where {UT <: Union{Cell, Union}} = begin
  if typeof(UT) != Union
    error("no method matching facet(::Type{$(UT)}) did you forget to define it?")
  end
  Union{facet(UT.a), facet(UT.b)}
end
#@Base.pure facet_count(::Type{Union{C, _}}) where {C <: Cell, _} = Union{facet(UT.a), facet(UT.b)}
@Base.pure face_count(::Type{C}, ::Type{C}) where {C <: Cell} = 1
@Base.pure facet_count(::Type{C}) where {C <: Cell} = face_count(C, facet(C))
@Base.pure function dim(::Type{UT}) where {UT <: Union{Cell, Union}}
  @assert dim(UT.a)==dim(UT.b)
  dim(UT.a)
end
@Base.pure dim_t(::Type{T}) where {T<:Cell} = Dim{dim(T)}()
@Base.pure complement(::Type{K}, ::Dim{i}) where {K <: Cell, i}= Codim{dim(K)-i}()
@Base.pure complement(::Type{K}, ::Codim{i}) where {K <: Cell, i} = Dim{dim(K)-i}()

################################################################################
# cell interface
################################################################################
cell_interface = [:dim, :vertex_count, :face_count, :facet, :facets, :volume, :reference_element]

for fn in cell_interface
  @eval $(fn)(::MethodNotImplemented) = throw("Method not implemented.")
end

################################################################################
# other cell related stuff
################################################################################

include("id.jl")
include("connectivity.jl")
include("geometry.jl")
include("generators.jl")

################################################################################
# cell id iterators
################################################################################
using TipiFEM.Utils: HeterogenousVector, HeterogenousIterator

const GenericIdIterator = AbstractVector{<:Id{<:Cell}}

# todo: with the 1.0 interface we can probably use Int for the type of the step in a range
const HomogeneousIdIterator{K <: Cell} = Union{OrdinalRange{Id{K}, Id{K}}, AbstractVector{Id{K}}}

const HeterogenousIdIterator = Union{HeterogenousVector{<:Id}, HeterogenousIterator{<:Id}}

const TypedIdIterator = Union{HomogeneousIdIterator, HeterogenousIdIterator}

const IdIterator = Union{GenericIdIterator, TypedIdIterator}

cell_type(::Union{T, Type{T}}) where T <: Union{HomogeneousIdIterator, HeterogenousIdIterator} = cell_type(eltype(T))

function IdIterator(::Union{Type{C}, C}) where C <: Cell
  if typeof(C) == Union
    HeterogenousVector{Tuple{uniontypes(C)...}}()
  elseif typeof(C) == DataType
    Vector{Id{C}}()
  else
    Vector{Id}()
  end
end
