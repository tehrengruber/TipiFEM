macro import_mesh_interface()
  quote
    import TipiFEM.Meshes: dim, coordinates, vertex_count, vertices, subcell,
                           connectivity, facet, face_count, facets, volume,
                           local_to_global, jacobian_transposed, jacobian_inverse_transposed,
                           reference_element, integration_element
  end
end

macro export_mesh_interface(cell_type)
  esc(quote
    # export mesh interface
    export dim, coordinates, vertex_count, vertices, boundary, subcell, volume, facet, face_count
    let cell_types = typeof($(cell_type)) <: AbstractArray ? $(cell_type) : subtypes($(cell_type))
      # export cell types
      for st in cell_types
        eval(:(export $(st.name.name)))
      end
      # call cell initializers
      for initializer in TipiFEM.Meshes.cell_initializer
        for st in cell_types
          initializer(st)
        end
      end
    end
  end)
end
