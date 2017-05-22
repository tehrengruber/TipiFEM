module Quadrature

using ComputedFieldTypes
using StaticArrays
using TipiFEM.Meshes: Cell, Geometry, MeshFunction, dim, cell_type, real_type,
                      integration_element, local_to_global
using TipiFEM.Utils: MethodNotImplemented

export Quadrule, integrate, integrate_local, quadrule

"""
Datatype storing a quadrature rule to evaluate integrals on mesh cells

 - `w`     - weights
 - `x`     - points
 - `scale` - scaling factors
"""
@computed type Quadrule{C <: Cell, order, REAL_T}
  w::SVector{order, REAL_T}
  x::SVector{order, SVector{dim(C), REAL_T}}
  scale::REAL_T
end

# todo: it would be nice if this dispatches only on mesh functions
#  with eltype <: Geometry
"""
Approximate integral of `f` over all cells in `mf`
"""
function integrate(f::Function, mf::MeshFunction, order=6)
  ∫ = (g) -> integrate(f, g, order=order)
  mapreduce(∫, +, mf)
end

"""
Appriximate integral of `f` over cell `geo` with order `order`
"""
function integrate{G <: Geometry}(f::Function, geo::G; order=25)
  integrate_local(x̂ -> f(local_to_global(geo, x̂)), geo)
  #w, x_local, s = quadrule(Quadrule{cell_type(G), 3, real_type(G)})
  ##let Φ = (x)->local_to_global(geo,x), δvol = (x)->integration_element(geo,x)
  #  sum = zero(real_type(G))
  #  @fastmath @inbounds @simd for i in 1:length(w)
  #    sum += w[i] * f(local_to_global(geo, x_local[i])) * integration_element(geo, x_local[i])
  #  end
  #  sum*=s
  #  #s * reduce(+, w .* f.(Φ.(x_local)) .* δvol.(x_local))
  ##end
end

"""
Integrate a function `f̂` that lives on the reference triangle over `geo`
"""
function integrate_local{G <: Geometry}(f̂::Function, geo::G; order=25)
  w, x_local, s = quadrule(Quadrule{cell_type(G), 12, real_type(G)})
  sum = zero(real_type(G))
  @fastmath @inbounds @simd for i in 1:length(w)
    sum += w[i] * f̂(x_local[i]) * integration_element(geo, local_to_global(geo, x_local[i]))
  end
  sum*=s
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
            nodes_t = SVector{order, SVector{dim(cell_type), real_type}},
            x = nodes_t(reinterpret(SVector{dim(cell_type), real_type}, x_us, (order,))...),
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
