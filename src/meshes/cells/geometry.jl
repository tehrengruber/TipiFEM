import StaticArrays.Size
import Base.getindex
using Base: @propagate_inbounds, @_propagate_inbounds_meta

@computed immutable Geometry{K <: Cell, world_dim, REAL_ <: Real} <: StaticMatrix{REAL_}
  data::NTuple{vertex_count(K)*world_dim, REAL_}

  function (::Type{Geometry{K, world_dim, REAL_}}){K <: Cell, world_dim, REAL_ <: Real, N}(in::NTuple{N, <:Real})
    new(in)
  end
end

@generated function (::Type{Geometry{K, world_dim, REAL_}})(xs::NTuple{N, SVector{world_dim, REAL_}}) where
    {K <: Cell, world_dim, REAL_ <: Real, N}
  # generate an expression that converts all given nodes into a matrix
  xs_mat_expr = Expr(:call, :hcat)
  for i in 1:N
    push!(xs_mat_expr.args, :(xs[$(i)]))
  end
  expr = quote
    # convert all nodes into matrix from
    @inbounds xs_mat = $(xs_mat_expr)
    # initialize geometry with nodes (needs to be transposed to match
    #  the storage format)
    xs_mat
    Geometry{K, world_dim, REAL_}(xs_mat')
  end
  expr
end

@Base.pure cell_type{G <: Geometry}(::Type{G}) = G.parameters[1]
@Base.pure world_dim{G <: Geometry}(::Type{G}) = G.parameters[2]
@Base.pure real_type{G <: Geometry}(::Type{G}) = G.parameters[3]

@Base.pure vertex_count{G <: Geometry}(::Type{G}) = vertex_count(cell_type(G))
@Base.pure vertex_count{G <: Geometry}(::G) = vertex_count(cell_type(G))
@Base.pure real_type{G <: Geometry}(::G) = G.parameters[3]

@Base.pure function Size{K <: Cell, world_dim, REAL_ <: Real, _}(G::Type{Geometry{K, world_dim, REAL_, _}})
  Size(vertex_count(cell_type(G)),world_dim)
end

@Base.pure function Size{K <: Cell, world_dim, REAL_ <: Real}(C::Type{Geometry{K, world_dim, REAL_}})
  Size(fulltype(C))
end

Base.@propagate_inbounds function getindex(geo::Geometry{K, world_dim, REAL_, _},
    i::Int) where{K <: Cell, world_dim, REAL_ <: Real, _}
  geo.data[i]
end

@propagate_inbounds @generated function point(geo::Geometry, i::Int)
  expr = Expr(:call, SVector{world_dim(geo), real_type(geo)})
  for j in 1:world_dim(geo)
    push!(expr.args, :(geo[i+$((j-1)*size(geo, 1))]))
  end
  quote
    @_propagate_inbounds_meta
    $(expr)
  end
end

reference_element(C::Type{<:Cell}) = reference_element(C())
