import StaticArrays.Size
import Base.getindex
using Base: @propagate_inbounds, @_propagate_inbounds_meta
using TipiFEM.Utils: tparam, @typeinfo

@computed struct Geometry{K <: Cell, world_dim, REAL_ <: Real} <: StaticMatrix{vertex_count(K), world_dim, REAL_}
  data::NTuple{vertex_count(K)*world_dim, REAL_}

  function (::Type{Geometry{K, world_dim, REAL_}})(in::NTuple{N, <:Real}) where {K <: Cell, world_dim, REAL_ <: Real, N}
    new(in)
  end
end

@generated function (::Type{Geometry{K, world_dim, REAL_}})(xs::NTuple{N, SVector{world_dim, <:Real}}) where
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

@generated function (::Type{Geometry{K, world_dim, REAL_}})(xs::NTuple{N, NTuple{world_dim, <:Real}}) where
    {K <: Cell, world_dim, REAL_ <: Real, N}
  # generate an expression that converts all given nodes into a matrix
  expr = Expr(:call, :(Geometry{K, world_dim, REAL_}))
  for i in 1:N
    push!(expr.args, :(SVector{world_dim, REAL_}(xs[$(i)])))
  end
  expr
end

@typeinfo cell_type(::Type{G}) where {G <: Geometry} = tparam(G, 1)
@typeinfo world_dim(::Type{G}) where {G <: Geometry} = tparam(G, 2)
@typeinfo real_type(::Type{G}) where {G <: Geometry} = tparam(G, 3)
@typeinfo vertex_count(::Type{G}) where {G <: Geometry} = vertex_count(cell_type(G))

@Base.pure function Size(G::Type{Geometry{K, world_dim, REAL_, _}}) where {K <: Cell, world_dim, REAL_ <: Real, _}
  Size(vertex_count(cell_type(G)),world_dim)
end

@Base.pure function Size(C::Type{Geometry{K, world_dim, REAL_}}) where {K <: Cell, world_dim, REAL_ <: Real}
  Size(fulltype(C))
end

Base.@propagate_inbounds function getindex(geo::Geometry{K, world_dim, REAL_, _},
    i::Int) where{K <: Cell, world_dim, REAL_ <: Real, _}
  geo.data[i]
end

@propagate_inbounds @generated function point(geo::Geometry, i::Int)
  if world_dim(geo) == 1 # in 1d we just return the a scalar
    @assert size(geo, 2)==1
    :(geo[i])
  else
    expr = Expr(:call, SVector{world_dim(geo), real_type(geo)})
    for j in 1:world_dim(geo)
      push!(expr.args, :(geo[i+$((j-1)*size(geo, 1))]))
    end
    quote
      @_propagate_inbounds_meta
      $(expr)
    end
  end
end

reference_element(C::Type{<:Cell}) = reference_element(C())
