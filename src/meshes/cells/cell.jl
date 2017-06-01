using TipiFEM.Utils: MethodNotImplemented

export @Polytope_str, Polytope, vertex_count, @Index_str, @Connectivity_str, vertex, Connectivity

################################################################################
# data-types
################################################################################
abstract type Cell end

################################################################################
# type-traits
################################################################################
# see https://en.wikipedia.org/wiki/N-skeleton
@Base.pure skeleton{T <: Cell}(::Type{T}) = dim(T) > 0 ? (skeleton(facet(T))..., T) : (T,)
@dim_dispatch @Base.pure skeleton{T <: Cell, d}(::Type{T}, ::Dim{d}) = skeleton(T)[1:d+1]
@Base.pure function subcell{T <: Cell, cd}(::Type{T}, ::Codim{cd})
  if cd == 0 # end recursion
    T
  elseif typeof(T) == Union
    Union{subcell(T.a, Codim{cd}()), subcell(T.a, Codim{cd}())}
  else
    subcell(facet(T), Codim{cd-1}())
  end
end
@Base.pure subcell{T <: Cell}(::Type{T}, d::Union{Int, Dim}) = subcell(T, Codim{dim(T)-convert(Int, d)}())
@Base.pure subcell{T <: Cell}(::Type{T}) = subcell(T, Codim{1}())
@Base.pure vertex{T <: Cell}(::Type{T}) = first(skeleton(T))
@Base.pure vertex_count{T <: Cell}(::Type{T}) = face_count(T, vertex(T))
@Base.pure facet{UT <: Union{Cell, Union}}(::Type{UT}) = begin
  if typeof(UT) != Union
    error("no method matching facet(::Type{$(UT)}) did you forget to define it?")
  end
  Union{facet(UT.a), facet(UT.b)}
end
#@Base.pure facet_count{C <: Cell, _}(::Type{Union{C, _}}) = Union{facet(UT.a), facet(UT.b)}
@Base.pure face_count{C <: Cell}(::Type{C}, ::Type{C}) = 1
@Base.pure facet_count{C <: Cell}(::Type{C}) = face_count(C, facet(C))
@Base.pure function dim{UT <: Union{Cell, Union}}(::Type{UT})
  assert(dim(UT.a)==dim(UT.b))
  dim(UT.a)
end
@Base.pure dim_t{T<:Cell}(::Type{T}) = Dim{dim(T)}()
@Base.pure complement{K <: Cell, i}(::Type{K}, ::Dim{i}) = Codim{dim(K)-i}()
@Base.pure complement{K <: Cell, i}(::Type{K}, ::Codim{i}) = Dim{dim(K)-i}()

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

include("index.jl")
include("connectivity.jl")
include("geometry.jl")

const IdIterator{K} = Union{AbstractVector{Index{K}}, Range{Index{K}}}
