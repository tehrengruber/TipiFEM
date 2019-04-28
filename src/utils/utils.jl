module Utils

import Base.Iterators.flatten
import Base: iterate

using MacroTools
using MacroTools: postwalk, prewalk

export flatten, type_scatter, type_gather, @sanitycheck, @apply_pure

struct MethodNotImplemented end

@generated function flatten(a::T) where T <: Union{Tuple, AbstractArray}
  if T <: Tuple
    expr = Expr(:tuple)
    for (i, TT) in enumerate(T.parameters)
      if TT <: Tuple || TT <: AbstractArray
        push!(expr.args, Expr(:..., :(flatten(a[$(i)]))))
      else
        push!(expr.args, :(a[$(i)]))
      end
    end
    expr
  elseif T <: AbstractArray
    error("not implemented")
  else
    :(a)
  end
end

macro apply_pure(expr)
  :(Base.Core._apply_pure($(expr.args[1]), $(Expr(:tuple, expr.args[2:end]...))))
end

"""
Given a tuple of types returns a tuple of types with all Union
types expanded as tuples
"""
type_scatter(t::Tuple) = map(type_scatter, t)
type_scatter(t::Type) = t
type_scatter(u::Union) = (Base.uniontypes(u)...,)

type_gather(t::Tuple) = map(e -> isa(e, Tuple) ? Union{e...} : e, t)

"""
Throw an error if the specified function is called and julia is set to be
in prototyping mode.

Todo: implement
"""
macro prototyping_only(fn_def)
  esc(fn_def)
end

const enable_sanity_checks = true

macro sanitycheck(expr)
  enable_sanity_checks ? esc(expr) : :()
end

"""
Given an expression return a canonical form of that expression

e.g. transform `a(x::T) where T <: Real = 1` into a{T}(x::T) = 1
"""
function canonicalize(expr::Expr)
  expr = longdef(expr)
  expr
#  postwalk(expr) do ex
#    if isa(ex, Expr) && ex.head == :where
#      @capture(ex, f_(args__) where Ts__)
#      Expr(:call, Expr(:curly, f, Ts...), args...)
#    else
#      ex
#    end
#  end
end

struct InvalidVecFnExprException <: Exception
  msg::String
  expr::Expr
end

function extract_return_types(body; error=error)
  return_types = []
  # find <<all return statements
  postwalk(body) do sub_expr
    if isa(sub_expr, Expr) && sub_expr.head != :block &&
        @capture(sub_expr, return (return_type_(args__) | return_expr_::return_type_))
      push!(return_types, return_type)
    end
    sub_expr
  end
  # the last statement is also a return statement
  let last_expr = body.args[end]
    assert(isa(last_expr, Expr))
    if isa(last_expr, Expr) && last_expr.head != :return
      @capture(last_expr, (return_type_(args__) | return_expr_::return_type_))
      push!(return_types, return_type)
    end
  end
  return_types
end

import Base.zeros
@generated function zero(::Type{NTuple{N, T}}) where {N, T <: Number}
  expr = Expr(:tuple)
  for i = 1:N
    push!(expr.args, 0)
  end
  expr
end

@Base.pure function tparam(T::Type, i::Int)
  isa(T, UnionAll) || isa(T, DataType) || error("$(T) is neither a DataType nor a UnionAll")
  # extract data type
  while !isa(T, DataType)
    T = T.body
  end
  # check that i is not a free type var
  isa(T.parameters[i], TypeVar) && error("the i-th tparam of $(T) is not bound")
  #
  T.parameters[i]
end

"""
Given a vectorized function definition generate an additional non vectorized
function definition. The vector width may be annotated by wrapping it into a
`@VectorWidth` macro call which is removed during expansion.

```@eval
expr=:(@generate_sisd a{num_points}(x::SMatrix{@VectorWidth(num_points), 2}) = x)
macroexpand(expr)
```

TODO: Currently this only works if the vectorized argument has been an SVector
resulting in an SMatrix. If the vectoized argument however has been a Scalar
an SVector would result, which is currently not supported by this macro.
"""
macro generate_sisd(expr)
  # helper function
  error = msg -> throw(InvalidVecFnExprException(msg, expr))
  #
  # search for the VecWidth argument
  #-(1-x̂[:, 2]),  -(1-x̂[:, 1])
  vector_width = nothing
  expr = prewalk(expr) do sub_expr
    if @capture(sub_expr, @VectorWidth(tmp_))
      vector_width = tmp
      sub_expr = vector_width
    end
    sub_expr
  end
  vector_width!=nothing || error("Vector width not found in expr: $(expr)")
  #
  # decompose expression
  #
  expr = macroexpand(__module__, expr) # expand macros
  expr = canonicalize(expr) # rewrite expression in a canonical form
  # ensure that the expression is a function definition
  expr.head ∈ (:function, :stagedfunction) || error("Expected function definition")
  sig = expr.args[1] # extract function signature
  body = expr.args[2] # extract function body
  body.head == :block || error("Unexpted syntax") # we take this for granted later on
  # decompose function signature
  @capture(sig, (f_(args__) where {Ts__}) | (f_(args__)::return_type_) where {Ts__}) || error("Expected function definition")
  # find all return types
  #  if the return type has been annotated in the signature use that type
  # otherwise search in the function body
  return_types = return_type == nothing ? extract_return_types(body, error=error) : [return_type]
  length(return_types)!=0 || error("Could not find return statement in $(expr)")
  length(return_types)==1 || error("Only a single return statement supported by now")
  #
  # parse expression
  #
  # remove vector width argument from parametric type arguments
  sisd_Ts=filter(Ts) do Targ
    @capture(Targ, T_ | (T_ <: B_)) || error("Unexpected syntax")
    T != vector_width
  end
  # remove :curly
  # rewrite arguments
  vector_args = Array{Bool, 1}(undef, length(args))
  sisd_args = map(1:length(args), args) do i, arg
    # rewrite vector arguments into scalar arguments
    # todo: allow function that do not specify T
    is_vector_arg = vector_args[i] = @capture(arg, x_::SMatrix{n_, m_, T_})
    if is_vector_arg && n == vector_width
      arg = :($(x)::SVector{$(m), $(T)})
    end
    # rewrite unlabeled arguments
    is_unlabeled = @capture(arg, ::T_)
    if is_unlabeled
      arg = Expr(:(::), gensym(), T)
    end
    arg
  end
  # extract argument labels
  forward_args = map(1:length(sisd_args), sisd_args) do i, arg
    @capture(arg, (x_::T_) | x_Symbol) || error("Unexpected syntax")
    if vector_args[i]
      @capture(arg, y_::SVector{m_, T_Symbol}) ||
        @capture(arg, y_::SVector{m_, T_Symbol <: TT_})
      @assert T != nothing "Unexpected syntax"
      x=:(convert(SMatrix{1, $(m), $(T)}, $(x)))
    end
    x
  end
  # generate call expression to the SIMD version
  forward_call_expr = Expr(:call, f, forward_args...)
  # generate expression that converts the return type from the SIMD into the SISD version
  call_expr = nothing
  for return_type in return_types
    @capture(return_type, T_{TT__} | T_) || error("Can not process return type $(return_type)")

    if T ∈ (:SArray, :SMatrix, :SVector)
      dims = if T == :SArray
        [TT[1]]
      elseif T == :SMatrix
        TT[1:2]
      else T == :SVector
        [TT[1]]
      end
      # search for the dimension(s) that contains the VectorWidth
      vector_dims = findall(dim -> dim==vector_width, dims)
      call_expr = Expr(:ref, forward_call_expr)
      for i in 1:length(dims)
        push!(call_expr.args, i ∈ vector_dims ? 1 : :(:))
      end
    else
      error("Can not process return type $(return_type)")
    end
  end
  sisd_expr = Expr(:function,
    Expr(:where, Expr(:call, f, sisd_args...), sisd_Ts...),
    call_expr
  )
  esc(Expr(:block, :(Base.@__doc__ $(expr)), sisd_expr))
end

macro typeinfo(expr)
  # rewrite expression in a canonical form
  expr = macroexpand(__module__, expr)
  expr = canonicalize(expr)
  # extract function name, signature, body from expression
  is_fn_def = @capture(expr, function f_(args__) where {targs__}
                   body_
                 end)
  if !is_fn_def
    targs=[]
    is_fn_def = @capture(expr, function f_(args__)
                                 body_
                               end)
  end
  is_fn_def || error("expected function definition")
  # generate function definition that dispatched on the value not on the type
  ntargs=copy(targs)
  nargs = map(args) do arg
    if @capture(arg, x_::Type{T_Symbol})
      :($(x)::$(T))
    elseif @capture(arg, ::Type{T_Symbol})
      :(::$(T))
    elseif @capture(arg, x_::Type{<:T_})
      new_targ = x
      push!(ntargs, Expr(:<:, new_targ, T))
      :(::$(new_targ))
    elseif @capture(arg, x_::Type{T_})
      error("argument $(arg) not supported by TipiFEM.Utils.@typeinfo")
    else
      arg
    end
  end
  nexpr = :(function $(f)($(nargs...)) where {$(ntargs...)}
              $(body.args...)
            end)
  # make both expression a pure function
  expr = :(Base.@__doc__ @Base.pure $(expr))
  nexpr = :(@Base.pure $(nexpr))
  # return the two expressions
  esc(Expr(:block, expr, nexpr))
end

#
# Index mapping
#
import Base: map, push!, values

mutable struct IndexMapping{I, V, II, VI}
  indices::II
  values::VI

  function (::Type{IndexMapping{I, V}})() where {I, V}
    let II = Vector{I}, VI = Vector{V}
      new{I, V, II, VI}(II(), VI())
    end
  end

  function (::Type{IndexMapping{I, V, II, VI}})(indices, values) where {I, V, II, VI}
    new{I, V, II, VI}(indices, values)
  end

  (::Type{IndexMapping{V}})() where {V} = IndexMapping{Int, V}()
end

IndexMapping(indices::II, values::VI) where {II, VI} = IndexMapping{eltype(II), eltype(VI), II, VI}(indices, values)

indices(indmap::IndexMapping) = indmap.indices
values(indmap::IndexMapping) = indmap.values

map(f, indmap::IndexMapping) = IndexMapping(indices(indmap), map(f, values(indmap)))

function push!(indmap::IndexMapping, i, v)
  push!(indices(indmap), i)
  push!(values(indmap), v)
end

function sort_by_indices!(indmap::IndexMapping)
  sort!(ZippedVector(indices(indmap), values(indmap)), by=x->x[1])
  indmap
end

#
# ZippedVector
#  contrary to zip a ZippedVector supports indexing operations
#
using ComputedFieldTypes
import Base: getindex, setindex!, eltype, iterate, size
import Base.Iterators: Zip
using Base: tuple_type_cons, tuple_type_head, tuple_type_tail

_zipped_vec_eltype(::Type{Is}) where {Is<:Tuple} =
    tuple_type_cons(eltype(tuple_type_head(Is)), _zipped_vec_eltype(tuple_type_tail(Is)))
_zipped_vec_eltype(::Type{Tuple{}}) = Tuple{}

@computed struct ZippedVector{T} <: AbstractVector{_zipped_vec_eltype(T)}
  is::T

  function ZippedVector(is::Vararg{<:AbstractVector})
    if !reduce(==, map(length, is))
      throw("Arguments to ZippedVector must be of equal length")
    end
    new{typeof(is)}(is)
  end
end

ZippedVector(is::Vararg{<:AbstractVector}) = ZippedVector{typeof(is)}(is...)

size(v::ZippedVector) = size(v.is[1])
eltype(::Type{ZippedVector{Is}}) where Is<:Tuple = _zipped_vec_eltype(Is)

@generated function getindex(v::ZippedVector{Is}, key) where Is <: Tuple
  expr = Expr(:tuple)
  for i in 1:length(Is.types)
    push!(expr.args, :(v.is[$(i)][key]))
  end
  expr
end

@generated function setindex!(v::ZippedVector{Is}, value, key) where Is <: Tuple
  expr = Expr(:block)
  for i in 1:length(Is.types)
    push!(expr.args, :(v.is[$(i)][key] = value[$(i)]))
  end
  expr
end

iterate(v::ZippedVector, state=1) = state > length(v) ? nothing : (v[state], state+1)

#
# Heterogenous vector
#
include("heterogenous_vector.jl")

#
# HeterogenousIterator
#
include("heterogenous_iterator.jl")
end
