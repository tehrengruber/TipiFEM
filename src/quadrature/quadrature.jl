module Quadrature

import Base.getindex

using ComputedFieldTypes
using StaticArrays
using TipiFEM.Meshes: Cell, Geometry, MeshFunction, dim, cell_type, real_type,
                      integration_element, local_to_global
using TipiFEM.Utils: MethodNotImplemented

export Quadrule, QuadruleList, integrate, integrate_local, quadrule

"""
Datatype storing a quadrature rule to evaluate integrals on mesh cells

 - `w`     - weights
 - `x`     - quadrature points
 - `scale` - scaling factors
"""
@computed struct Quadrule{C <: Cell, order, REAL_T}
  w::SVector{order, REAL_T}
  x::SVector{order, dim(C) == 1 ? REAL_T : SVector{dim(C), REAL_T}}
  scale::REAL_T
end

struct QuadruleList{Cs <: Tuple, RULES_T <: Tuple}
  quadrules::RULES_T
end

function (::Type{QuadruleList{Cs}})(quadrules::RULES_T) where {Cs <: Tuple, RULES_T <: Tuple}
  QuadruleList{Cs, RULES_T}(quadrules)
end

@generated function QuadruleList(rules_t::Type...)
  cell_type_expr = Expr(:curly, Tuple)
  data_exprs = Expr(:tuple)
  for rule_t in map(T -> first(T.parameters), rules_t)
    push!(cell_type_expr.args, tparam(rule_t, 1))
    push!(data_exprs.args, :(quadrule($(rule_t))))
  end
  :(QuadruleList{$(cell_type_expr)}($(data_exprs)))
end

@generated function getindex(list::QuadruleList{Cs}, ::Type{C}) where {Cs, C <: Cell}
  i = findfirst(isequal(C), Cs.parameters)
  i != 0 || error("No quadrule for cell type $(C) in list")
  :(list.quadrules[$(i)])
end

# todo: it would be nice if this dispatches only on mesh functions
#  with eltype <: Geometry

using TipiFEM.Utils: tparam

# todo: slow!
"""
Integrate a function `f̂` that lives on the reference element over K̂

  ∫_K̂ f̂(x̂) dx̂

"""
function integrate_local(f̂::Function, ::K; rules=nothing) where K <: Cell
  ωs, x̂s, s = rules[K]

  s*mapreduce((ω, x̂) -> ω * f̂(x̂), +, ωs, x̂s)
end

"""
Integrate a function `f̂` that lives on the reference element over K

  ∫_K f̂(Φ⁻¹(x)) dx = ∫_K̂ f̂(x̂) |det(DΦ(x̂))| dx̂

"""
function integrate_local(f̂::Function, geo::Geometry{K}; rules=nothing) where K <: Cell
  detDΦ(x̂) = integration_element(geo, x̂)
  f̂′(x̂) = f̂(x̂)*detDΦ(x̂) # integrand

  integrate_local(f̂′, K(); rules=rules)
end

function integrate(f::Function, geo::Geometry{K}; rules=nothing) where K <: Cell
  Φ(x̂) = local_to_global(geo, x̂)
  detDΦ(x̂) = integration_element(geo, x̂)
  f̂(x̂) = f(Φ(x̂))*detDΦ(x̂) # integrand

  integrate_local(f̂, K(); rules=rules)
end

# access to quadrature rule data is given by means of calling this function
#  which is implemented in the mesh implementation
quadrule(::MethodNotImplemented) = error("method not implemented")

"""
Generate quadrature rules for `cell_type` using `real_type` precision arithmetic
and quadrature data `quadrules`
"""
macro generate_quadrature_rules(cell_type, real_type, quadrules)
  esc(quote
    using TipiFEM.Meshes: dim
    using TipiFEM.Quadrature: Quadrule
    import TipiFEM.Quadrature: quadrule

    let cell_type=$(cell_type), real_type = $(real_type)
      for (order, quadrule_data) in $(quadrules)
        let (w, x_us, s)=quadrule_data, # extract quad_data
            node_t = dim(cell_type) == 1 ? real_type : SVector{dim(cell_type), real_type},
            nodes_t = SVector{order, node_t},
            x = nodes_t(reinterpret(node_t, x_us)...),
            quadrule = Quadrule{cell_type, order, real_type}(w, x, s)
          eval(:(@Base.pure function quadrule(::Type{Quadrule{$(cell_type), $(order), $(real_type)}})
            $(quadrule.w)::$(typeof(quadrule.w)), $(quadrule.x)::$(typeof(quadrule.x)), $(quadrule.scale)::$(typeof(quadrule.scale))
          end))
        end
      end
    end
  end)
end

end
