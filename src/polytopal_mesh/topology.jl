#import PolytopalMesh.initialize_connectivity

function initialize_connectivity(mesh, ::Type{Polytope"3-node triangle"})
  # order edges
  #perm = sortperm(connectivity(mesh, mesh_dim(mesh), 0).values, lt=(c1, c2)->c1[1]<c2[1])
  for conn in connectivity(mesh, Dim{mesh_dim(mesh)}(), Dim{0}())
    println((conn))
    for edge_conn in facets(conn)
      add_cell!(mesh, Edge, edge_conn)
    end
  end
end
