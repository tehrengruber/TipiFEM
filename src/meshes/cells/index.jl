import Base.promote_rule

using Base: bitcast
using TipiFEM.Utils: @prototyping_only

@prototyping_only LocalDOFIndex(idx::Int) = LocalDOFIndex{idx}()

 # todo: currently an index is a subtype of an Integer to enable usage of
 #  Base.OneTo. Indices are however not purely Integers and should be therefore
 #  not a subtype of Integer (this is similar to why Pointers are no Integers).
 #
primitive type Id{K <: Cell} <: Integer 64 end

Id{K}(id::Int) where K <: Cell = reinterpret(Id{K}, id)

import Base.convert
function convert(::Type{Id{K}}, idx::Int) where {K <: Cell}
  #typeof(K)==DataType || error("Cell type of an index must be concrete")
  #assert(typeof(K)==DataType)
  bitcast(Id{K}, idx)
end

import Base: show

function show(io::IO, id::Id)
  show(io, typeof(id))
  write(io, "(")
  show(io, convert(Int, id))
  write(io, ")")
end

@Base.pure cell_type(::Id{K}) where {K <: Cell} = K
@Base.pure cell_type(::Type{Id{K}}) where {K <: Cell} = K

@Base.pure function cell_type(T::Type{Union{Id{K}, _}}) where {K <: Cell, _}
  Union{cell_types(T)...}
end

@Base.pure function cell_types(T::Type{IDX}) where {IDX <: Id}
  (map(T -> cell_type(T), Base.uniontypes(T))...,)
end

@inline convert(::Type{Int}, i::IDX) where {IDX <: Id} = bitcast(Int, i)

promote_rule(::Type{IDX}, ::Type{Int}) where {IDX <: Id} = IDX
promote_rule(::Type{IDX1}, ::Type{IDX2}) where {IDX1 <: Id, IDX2 <: Id} = Id

# logic operations on indices
for op in (:<, :>, :<=, :>=, :%)
    @eval begin
			import Base.$(op)
			$(op)(i1::IDX, i2::IDX) where {IDX <: Id} = $(op)(convert(Int, i1), convert(Int, i2))
		end
end
# arithmetic operations on indices
#  *(::Id, ::Id) is not so nice, but needed to use repeat.
#  maybe a seperate repeat iterator solves this
for op in (:+, :-, :*)
    @eval begin
			import Base.$(op)
			$(op)(i1::IDX, i2::IDX) where {IDX <: Id} = $(op)(convert(Int, i1), convert(Int, i2))
		end
end

import Base.one
one(::Type{IDX}) where {IDX <: Id} = IDX(1)

import Base.hash
hash(i::Id) = hash(convert(Int, i))

import Base.unitrange_last
# todo: create julia pull request for this, but implement it in the constructor
#  of a unitrange
#unitrange_last(start::Integer, stop::Integer) = unitrange_last(promote(start, stop)...)

# fix start of OneTo ranges
#import Base: start, length, OneTo, step
#start(r::OneTo{Id{C}}) where C <: Cell = 1

# fix length on ranges of indices (otherwise an Id is returned)
#import Base.length
#for T in (UnitRange, OneTo)
#  @eval length(r::$(T){Id{C}}) where C <: Cell = convert(Int, last(r)-first(r))+1
#end

#length(r::StepRange{Id{C}}) where C <: Cell = div(convert(Int, r.stop-r.start), step(r))+1

#step(r::StepRange{Id{C}}) where C <: Cell = convert(Int, r.step)

# this is used for sorting
import Base.sub_with_overflow
sub_with_overflow(x::Id, y::Id) = sub_with_overflow(convert(Int, x), convert(Int, y))
