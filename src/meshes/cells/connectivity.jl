import StaticArrays.Size
import Base.getindex

# we denote the two types of objects of a incidence relation as the incidenter type
#  and the incidentee type.
# todo: look in the literature whether a different (maybe more meaningful)
#  naming exists.

# todo: a more precise statement can be made if the mesh type is known
#  - the iteratorsize for a regular mesh for example always has a length
#  - the iteratorsize for a mesh with bounded vertex out degree is always bounded
"""
Given two cell types returns one of the following values:
- HasLength() if the number of cells incident to cells of type `T1` is fixed
- SizeUnknown() if the number of cells incident to cells of type `T2` cannot be determined in advance
"""
function incidentee_iteratorsize{T1 <: Cell, T2 <: Cell}(::Type{T1}, ::Type{T2})
  dim(T1) >= dim(T2) ? Base.HasLength() : Base.SizeUnknown()
end

"""
Given two cell types `T1`, `T2` where the number of cells incident to cells of
type `T1` is fixed return the number of incident cells.
"""
@Base.pure function incidentee_count{T1 <: Cell, T2 <: Cell}(::Type{T1}, ::Type{T2})
  # note: if this function is not declared pure setindex on an array with
  #  eltype Connectivity{...} and value a tuple (i.e. a value that must
  #  be converted to eltype) will throw strange errors, because :
  #  ERROR: TypeError: resize!: in typeassert, expected Connectivity{K, K, 3}
  #    got Connectivity{K, K,3}
  !(dim(T1) == 0 && dim(T2) == 0) || error("incidence relation between vertices is not defined")
  if dim(T1) == dim(T2)
    face_count(T1, subcell(T2))
  elseif dim(T1) > dim(T2)
    face_count(T1, T2)
  else
    error("incidentee count is not known given only type information")
    -1
  end
end

#
# Connectivity
#
"Store all indices of `T2` cells incident to a `T1` cell."
@computed immutable Connectivity{T1 <: Cell, T2 <: Cell} <: StaticVector{Index{T2}}
    data::NTuple{incidentee_count(T1, T2), Index{T2}}

    function (::Type{Connectivity{T1, T2}}){T1 <: Cell, T2 <: Cell, N, IT}(in::NTuple{N, IT})
      @boundscheck all(x->x!=0, in) || error("Cell ids must be non zero")
      new(in)
    end
end

function (::Type{Connectivity{T1, T2}}){T1 <: Cell, T2 <: Cell}()
  Connectivity{T1, T2}((-1 for i in 1:incidentee_count(T1, T2))...)::fulltype(Connectivity{T1, T2})
end

incidenter_type{T1 <: Cell, T2 <: Cell}(::Type{Connectivity{T1, T2}}) = T1

incidentee_type{T1 <: Cell, T2 <: Cell}(::Type{Connectivity{T1, T2}}) = T2

incidenter_type{T1 <: Cell, T2 <: Cell, _}(::Type{Connectivity{T1, T2, _}}) = T1

incidentee_type{T1 <: Cell, T2 <: Cell, _}(::Type{Connectivity{T1, T2, _}}) = T2

@Base.pure function Size{T1, T2, _}(C::Type{Connectivity{T1, T2, _}})
  Size(incidentee_count(T1, T2))
end

@Base.pure function Size{T1, T2}(C::Type{Connectivity{T1, T2}})
  Size(incidentee_count(T1, T2))
end

function vertices(c::Connectivity)
  assert(incidentee_type(c) == subcell(incidenter_type(c), Dim{0}()))
  c.data
end

import Base: isless, reverse
"""
A connectivity object is less then another connectivity object if all indices
of the incident objects of the first connectivity object are smaller then
indices of the second connectivity object.
"""
isless(c1::Connectivity, c2::Connectivity) = isless(c1.data, c2.data)
reverse(c::C) where C <: Connectivity = C(reverse(c.data))

"Retrieve index of the i-th facet incident to the cell `v` belongs to"
Base.@propagate_inbounds getindex(v::Connectivity, i::Int) = v.data[i]

"Get the index of the `i`th vertex of a cell of type K."
function vertex{K <: Cell}(conn::Connectivity{K}, i::Int)
  @assert dim(incidentee_type(typeof(conn))) == 0
  conn[i]
end

import Base.show
function show(io::IO, conn::Connectivity)
  write(io, "$(eltype(conn))$(map(idx->convert(Int, idx), conn))")
end

const NeighbourConnectivity{C <: Cell} = Connectivity{C, C}

#
# VariableConnectivity
#
const VariableConnectivity{FACE <: Cell} = Array{Index{FACE}, 1} # K <: Cell,

#
# NeighbourConnectivity
#
#"Stores incident facets of type FT to a cell of type CT"
#@computed immutable NeighbourConnectivity{K <: Cell} <: StaticVector{Index{K}}
#    data::NTuple{face_count(K, subcell(K)), Index{K}}
#
#    function (::Type{NeighbourConnectivity{K}}){K <: Cell, N, IT}(in::NTuple{N, IT})
#      new(in)
#    end
#end
#
#@Base.pure function Size{K, _}(C::Type{NeighbourConnectivity{K, _}})
#  Size(face_count(cell_type(C), subcell(C)),)
#end
#
#@Base.pure function Size{K}(C::Type{NeighbourConnectivity{K}})
#  Size(fulltype(C))
#end
