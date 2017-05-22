import Base.promote_rule,
       Base.convert

immutable Codim{i} <: Number end
immutable Dim{i} <: Number end

convert{i}(::Type{Int}, ::Codim{i}) = i
convert{i}(::Type{Int}, ::Dim{i})   = i

promote_rule{d}(::Type{Dim{d}}, ::Type{Int}) = Dim{d}

import Base.-
@generated function -{d, cd}(::Dim{d}, ::Codim{cd})
  :($(Dim{d-cd})())
end

## arithmetic operations
#for op in (:+, :-, :<, :>, :<=, :>=)
#    @eval begin
#      import Base.$(op)
#      # codim
#      #$(op){CODIM_ <: Codim}(a::CODIM_, b::Int) = Codim{$(op)(convert(Int, a), b)}()
#      #$(op){CODIM_ <: Codim}(a::Int, b::CODIM_) = Codim{$(op)(a, convert(Int, b))}()
#      # Dim
#      $(op){DIM_ <: Dim}(::Dim{i}, j::Int) = Dim{$(op)(convert(Int, a), b)}()
#      $(op){DIM_ <: Dim}(a::Int, ::Dim{j}) = Dim{$(op)(a, convert(Int, b))}()
#    end
#end

for op in (:+, :-, :<, :>, :<=, :>=, :(==))
    @eval begin
      import Base.$(op)
      $(op){i, j}(a::Dim{i}, b::Dim{j}) = $(op)(i, j)
      $(op){i, j}(a::Codim{i}, b::Codim{j}) = $(op)(i, j)
    end
end

## convenience macro for functions that dispatch on types
# helper function
function is_dim_arg(arg)
  isa(arg, Symbol) && return false
  if arg.head == :kw
    arg = arg.args[1]
  end
  isa(arg, Symbol) && return false
  t = if arg.head == :(::)
    last(arg.args)
  else
    error("unexpected expression $(arg). please submit a bug report.")
  end
  if isa(t, Symbol)
    t ∈ (:Dim, :Codim)
  elseif t.head == :curly
    first(t.args) ∈ (:Dim, :Codim)
  else
    error("unexpected expression. please submit a bug report.")
  end
end

"""
Given a function expression with parametric type `K`, with K a Codim 0 cell,
that takes arguments of type Dim or Codim generates additional functions
that allow calling with dimensional arguments of different type.
Therefore a function that originally takes Dim arguments may be called with Codim
arguments.

Example:

```
# original function
@dim_dispatch hasindex{K <: Cell, i, j}(::Type{K}, ::Dim{j}, ::Dim{j}) = nothing
# functions generated by this macro
hasindex{K <: Cell, i, j}(topology::MeshTopology{K}, ::Codim{j}, ::Dim{j}) = hasindex(topology, complement(K, Codim{j}()), Dim{j}())
hasindex{K <: Cell, i, j}(topology::MeshTopology{K}, ::Dim{j}, ::Codim{j}) = hasindex(topology, Dim{j}(), complement(K, Codim{j}()))
hasindex{K <: Cell, i, j}(topology::MeshTopology{K}, ::Codim{j}, ::Codim{j}) = hasindex(topology, complement(K, Codim{j}()), complement(K, Codim{j}()))
```
"""
macro dim_dispatch(fn_def)
  #
  # preprocessing
  #
  # expand all macro
  fn_def = macroexpand(fn_def)
  # canonicalize notation
  if fn_def.head == :(=)
    assert(length(fn_def.args)==2)
    fn_def = Expr(:function, fn_def.args[1], fn_def.args[2])
  end
  #
  fn_def.head == :function || fn_def.head == :stagedfunction || error("Expression must be a function definition")
  fn_sig = first(fn_def.args)
  # create a BitVector that is true for every argument that is either a Codim or a Dim
  dim_args = map(is_dim_arg, fn_sig.args[2:end])
  any(dim_args) || error("Could not find any (co-)dimension arguments in function signature $(fn_sig)")
  dim_args_indices = map(x->x[1], filter(x->x[2], collect(enumerate(dim_args))))
  n = length(filter(x->x==true, dim_args))
  named_dim_args = map(fn_sig.args[2:end]) do arg
    is_dim_arg(arg) && length(arg.args)==2
  end
  exprs = Expr(:block, :(Base.@__doc__ $(fn_def)))
  # create prototype signature for all permutations by naming non dimensional arguments
  #  which are not named yet
  perm_fun_sig_proto = copy(fn_sig)
  perm_fun_args_proto = []
  for (i, arg) in enumerate(perm_fun_sig_proto.args[2:end])
    # if the argument has a default value we just omit the value
    if isa(arg, Expr) && arg.head==:kw
      arg = arg.args[1]
    end
    if typeof(arg) == Symbol # if the argument is untyped
      push!(perm_fun_args_proto, arg)
    elseif is_dim_arg(arg)
      push!(perm_fun_args_proto, nothing) # these are ignored because they are handled later anyway
    elseif arg.head == :(::) && length(arg.args)==1
      name = gensym()
      arg.args=[name, arg.args[1]] # here we change perm_fun_sig_proto
      push!(perm_fun_args_proto, name)
    elseif arg.head == :(::)
      push!(perm_fun_args_proto, arg.args[1])
    else
      error("unexpected argument signature")
    end
  end
  # create all permutations
  permutations = []
  for perm_idx in 0:2^n-1
    perm_fn_sig = copy(perm_fun_sig_proto)
    perm_fn_body = Expr(:call, isa(perm_fn_sig.args[1], Symbol) ? perm_fn_sig.args[1] : perm_fn_sig.args[1].args[1], perm_fun_args_proto...)
    # extract permutation from perm_idx
    perm = [(perm_idx&2^i)==0 ? :Dim : :Codim for i in 0:n-1]
    # alter function signature
    skip=true
    for (i, p) in zip(dim_args_indices, perm)
      arg_sig = copy(perm_fn_sig.args[i+1])
      assert(arg_sig.head == :(::))
      # modify argument signature (something like v::Dim)
      if isa(arg_sig.args[end], Expr) && arg_sig.args[end].head == :curly
        arg_sig.args[end].args[1] = p
      else
        arg_sig.args[end] = p
      end
      arg = if named_dim_args[i]
        arg_sig.args[1]
      else
        arg = :($(arg_sig.args[end])())
      end
      changed = perm_fn_sig.args[i+1] != arg_sig
      skip = skip && !changed
      perm_fn_sig.args[i+1] = arg_sig
      perm_fn_body.args[i+1] = changed ? :(complement(K, $(arg))) : arg
    end
    if !skip
      push!(exprs.args, :($(perm_fn_sig) = $(perm_fn_body)))
    end
  end
  esc(exprs)
end
