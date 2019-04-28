import Base: length, size, eltype, map, mapfoldl, zip, collect, map, zip,
             collect, mapreduce, mapfoldl

# Concatenate the output of n iterators
struct HeterogenousIterator{T, Ts <: Tuple}
  iters::Ts
end

@generated function (::Type{HeterogenousIterator{Ts}})(data) where {Ts <: Tuple}
  :(HeterogenousIterator{$(Union{map(T -> eltype(T), Ts.parameters)...}), Ts}(data))
end

function HeterogenousIterator(data::Ts) where {Ts <: Tuple}
  HeterogenousIterator{Ts}(data)
end

decompose(hiter::HeterogenousIterator) = hiter.iters

iteratorsize(::Type{HeterogenousIterator{T, Ts}}) where {T, Ts} = _het_it_is(Ts)

@generated function _het_it_is(t::Type{Ts}) where {Ts}
    for itype in Ts.types
        if iteratorsize(itype) == IsInfinite()
            return :(IsInfinite())
        elseif iteratorsize(itype) == SizeUnknown()
            return :(SizeUnknown())
        end
    end
    return :(HasLength())
end

@generated function compose(iters...)
  if any(T->T<:HeterogenousIterator, iters)
    # expand inner heterogenous iterators
    types = []
    expr = Expr(:tuple)
    for (i, IT) in zip(1:length(iters),iters)
      if IT <: Chain
        push!(types, IT.parameters[1].types...)
        push!(expr.args, Expr(:..., :(iters[$(i)].iters)))
      else
        push!(types, IT)
        push!(expr.args, :(iters[$(i)]))
      end
    end
    return :(HeterogenousIterator($(expr)))
  end

  :(HeterogenousIterator(iters))
end

length(it::HeterogenousIterator{Tuple{}}) = 0
length(it::HeterogenousIterator) = sum(length, it.iters)
size(it::HeterogenousIterator) = (length(it),)

#eltype{T}(::Type{Chain{T}}) = typejoin([eltype(t) for t in T.parameters]...)
@Base.pure eltype(::Type{HeterogenousIterator{T, Ts}}) where {T, Ts} = Union{(eltype(t) for t in Ts.types)...}

function iterate(it::HeterogenousIterator)
  for i in 1:length(it.iters)
      val, it_state = iterate(it.iters[i])
      if it_state != nothing
        return (val, (i, it_state))
      end
  end
  nothing
end

function iterate(it::HeterogenousIterator, state)
  i0, it_state = state
  # check if there are elements left in the current (sub)iterator
  let it_result = iterate(it.iters[i0], it_state)
    if it_result != nothing
      val, it_state = it_result
      return (val, (i0, it_state))
    end
  end
  # check remaining (sub)iterators
  for i in i0+1:length(it.iters)
      it_result = iterate(it.iters[i])
      if it_result != nothing
        val, it_state = it_result
        return (val, (i, it_state))
      end
  end
  nothing
end

map(f, it::HeterogenousIterator) = compose(map(iter -> map(f, iter), it.iters))
mapfoldl(f, op, it::HeterogenousIterator) = reduce(op, (mapreduce(f, op, link) for link in it.iters))
mapfoldl(f, op, v0, it::HeterogenousIterator) = reduce(op, v0, (mapreduce(f, op, v0, link) for link in it.iters))
zip(it1::HeterogenousIterator, it2::HeterogenousIterator) = compose(map((link1, link2) -> zip(link1, link2), it1.iters, it2.iters))
collect(it::HeterogenousIterator) = compose(map(link -> collect(link), it.iters))
foreach(f, it::HeterogenousIterator) = foreach(iter -> foreach(f, iter), it.iters)
