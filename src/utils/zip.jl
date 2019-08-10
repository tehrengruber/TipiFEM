using Base: @propagate_inbounds

import Base.Iterators: iterate, length, eltype, axes, size, IteratorSize, IteratorEltype

struct Zip2{Its <: Tuple}
  its::Its

  function Zip2(its::Its) where {Its <: Tuple}
    @boundscheck begin
      l = length(its[1])
      for i in 2:length(its)
        @assert l == length(its[i])
      end
    end
    new{Its}(its)
  end
end

@propagate_inbounds zip2(its...) = Zip2(its)
length(z::Zip2) = length(first(z.its))
size(z::Zip2) = (length(z),)
axes(z::Zip2, d) = OneTo(d==1 ? length(z) : 1)
eltype(z::Zip2{Its}) where Its <: Tuple = Base.Iterators._zip_eltype(Its)

IteratorSize(::Type{Zip2{Its}}) where Its <: Tuple = IteratorSize(Zip{Its})
IteratorEltype(::Type{Zip2{Its}}) where Its <: Tuple = IteratorEltype(Zip{Its})

@propagate_inbounds @generated function iterate(z::Zip2, state=nothing)
  init = (state == Nothing)
  n = length(tparam(z, :Its).parameters)
  vs = [Symbol("v$(i)") for i in 1:n]

  ex = Expr(:block)
  for i in 1:n
    iter_ex = init ? :(iterate(z.its[$(i)])) : :(iterate(z.its[$(i)], state[$(i)]))
    push!(ex.args, quote
      $(vs[i]) = $(iter_ex)
      $(vs[i]) === nothing && return nothing
    end)
  end
  reex = Expr(:tuple)
  rsex = Expr(:tuple)
  for i in 1:n
    push!(reex.args, :($(vs[i])[1]))
    push!(rsex.args, :($(vs[i])[2]))
  end
  push!(ex.args, :(($(reex), $(rsex))))
  ex
end
