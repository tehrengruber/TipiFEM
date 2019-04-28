using ComputedFieldTypes

import Base: length, size, push!, getindex, foreach, map, map!, eltype

export HeterogenousVector, compose, decompose

_hetarrhelper = (Ts) -> Tuple{map(T -> Vector{T}, Ts.parameters)...}

@computed struct HeterogenousVector{T, Ts <: Tuple} <: AbstractArray{T, 1}
  data::_hetarrhelper(Ts)
end

function (::Type{HeterogenousVector{Ts}})(data) where {Ts <: Tuple}
  HeterogenousVector{Union{Ts.parameters...}, Ts}(data)
end

@generated function HeterogenousVector(data::Ts) where {Ts <: Tuple}
  :(HeterogenousVector{$(Tuple{map(T -> eltype(T), Ts.types)...})}(data))
end

function (::Type{HeterogenousVector{Ts}})() where {Ts <: Tuple}
  HeterogenousVector{Union{Ts.parameters...}, Ts}(map(T -> Vector{T}(), (Ts.parameters...)))
end

"""
Return the array that contains the elements of type `ET`
"""
@generated function getindex(arr::HeterogenousVector{T, Ts}, ::Type{Ti}) where {T, Ts, Ti}
  i = findfirst(Ts.parameters, Ti)
  if i==0
    error("HeterogenousArray has no subarray with element type $(ET)")
  end
  :(arr.data[$(i)])
end

eltype(arr::HeterogenousVector{T}) where T = T
size(arr::HeterogenousVector) = (length(arr),)
length(arr::HeterogenousVector) = mapreduce(length, +, decompose(arr))
push!(arr::HeterogenousVector, v::T) where {T} = push!(arr[T], v)

function getindex(arr::HeterogenousVector, j::Int)
  offset = 0
  i=0
  while offset<j
    i+=1
    offset += length(arr.data[i])
  end
  arr.data[i][j-offset+length(arr.data[i])]
end

decompose(vec::Vector) = (vec,)

decompose(arr::HeterogenousVector) = arr.data
compose(arrs::Vector...) = HeterogenousVector(arrs)
@generated function compose(arrs::Tuple)
  expr = Expr(:call, :compose)
  for i in 1:length(arrs.types)
    push!(expr.args, :(arrs[$(i)]))
  end
  expr
end

foreach(f::Function, harr::HeterogenousVector) = foreach(arr -> foreach(f, arr), decompose(harr))
map(f::Function, harr::HeterogenousVector) = compose(map(arr -> map(f, arr), decompose(harr)))
map!(f::Function, harr::HeterogenousVector) = compose(map!(arr -> map(f, arr), decompose(harr)))
