macro import_mesh_interface()
  quote
    import TipiFEM.Meshes: dim, vertex_count, vertices, subcell,
                           connectivity, facet, face_count, facets, volume,
                           local_to_global, jacobian_transposed, jacobian_inverse_transposed,
                           reference_element, integration_element, canonicalize_connectivity,
                           flip_orientation, is_cannonical, global_to_local, midpoint
  end
end

macro export_mesh_interface(cell_type)
  esc(quote
    using InteractiveUtils: subtypes

    # export mesh interface
    export dim, vertex_count, vertices, boundary, subcell, volume, facet, face_count
    let cell_types = typeof($(cell_type)) <: AbstractArray ? $(cell_type) : subtypes($(cell_type))
      # export cell types
      for st in cell_types
        if !isa(st, Union)
          eval(:(export $(st.name.name)))
        end
      end
      # call cell initializers
      append!(TipiFEM.Meshes.registered_cell_types, cell_types)
      for (initializer, attributes) in TipiFEM.Meshes.cell_initializer
        for st in cell_types
          # ignore initializers for hybrid cells
          if isa(st, Union) && !attributes[:hybrid]
            continue
          end
          initializer(@__MODULE__, st)
        end
      end
    end
  end)
end
