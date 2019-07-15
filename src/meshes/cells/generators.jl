"""
Generate face_count(::Type{T}, ::Type{...}) methods for a cell T
"""
function generate_face_count_methods(mod, ::Type{T}) where T <: Cell
  dim(T) != 0 || return # vertices have no faces
  expr = Expr(:block)
  push!(expr.args, :(import TipiFEM.Meshes: face_count))
  if dim(T) == 1
    push!(expr.args, quote
      face_count(::Type{$(T)}, ::Type{$(vertex(T))}) = $(vertex_count(T))
    end)
  elseif dim(T) >= 1
    facets_info = facets(T)
    facet_types = unique(map(f -> f[1], facets_info))
    for face_type in facet_types
      face_count = length(collect(Base.Iterators.filter(face_info -> face_info[1] == face_type, facets_info)))
      push!(expr.args, quote
        face_count(::Type{$(T)}, ::Type{$(face_type)}) = $(face_count)
      end)
    end
  else
    error("Unexpected dimension. $(T) has dimension $(dim(T))")
  end
  mod.eval(expr)
end

"""
Generate facet(::Connectivity{T, ...}) and facet(::Geometry{T, ...}) methods
"""
function generate_facets_methods(mod, ::Type{T}) where T <: Cell
  dim(T) != 0 || return # vertices have no facets
  exprs = Expr(:block)
  push!(exprs.args, :(import TipiFEM.Meshes: facets))
  if dim(T) == 1
    push!(exprs.args, :(facets(cell::Connectivity{$(T), $(vertex(T))}) = cell.data))
  elseif dim(T) >= 1
    facets_info = facets(T)
    # generate facet(::Connectivity{...}) methods
    let body_expr = Expr(:tuple)
      for facet_info in facets_info
        facet_type = facet_info[1]
        facet_vertex_indices = facet_info[2]
        expr = Expr(:call, :(Connectivity{$(facet_type), $(vertex(T))}))
        for facet_vertex_index in facet_vertex_indices
          push!(expr.args, :(vertex(cell, $(facet_vertex_index))))
        end
        push!(body_expr.args, expr)
      end
      push!(exprs.args, quote
        function facets(cell::Connectivity{$(T), $(vertex(T))})
          $(body_expr)
        end
      end)
    end
    # generate facet(::Geometry{...}) methods
    let body_expr = Expr(:tuple)
      for facet_info in facets_info
        facet_type = facet_info[1]
        facet_vertex_indices = facet_info[2]
        expr = Expr(:call, :(Geometry{$(facet_type), world_dim(G), real_type(G)}))
        for facet_vertex_index in facet_vertex_indices
          push!(expr.args, :(point(cell, $(facet_vertex_index))))
        end
        push!(body_expr.args, expr)
      end
      push!(exprs.args, quote
        function facets(cell::G) where G <: Geometry{$(T)}
          $(body_expr)
        end
      end)
    end
  else
    error("Unexpected dimension. $(T) has dimension $(dim(T))")
  end
  mod.eval(exprs)
end

# add cell initializers
add_cell_initializer(generate_face_count_methods)
add_cell_initializer(generate_facets_methods)
