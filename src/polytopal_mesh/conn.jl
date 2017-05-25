facet(::Type{Polytope"2-node line"}) = Polytope"1-node point"
facet(::Type{Polytope"3-node triangle"}) = Polytope"2-node line"
facet(::Type{Polytope"4-node quadrangle"}) = Polytope"2-node line"

face_count(::Type{Polytope"3-node triangle"}, ::Type{Polytope"2-node line"}) = 3
face_count(::Type{Polytope"4-node quadrangle"}, ::Type{Polytope"2-node line"}) = 4

function facets(tri::Connectivity{Polytope"3-node triangle", Vertex})
  (Connectivity{Edge, Vertex}(vertex(tri, 1), vertex(tri, 2)),
   Connectivity{Edge, Vertex}(vertex(tri, 2), vertex(tri, 3)),
   Connectivity{Edge, Vertex}(vertex(tri, 3), vertex(tri, 1)))
end

function facets(quad::Connectivity{Polytope"4-node quadrangle", Vertex})
  (Connectivity{Edge, Vertex}(vertex(quad, 1), vertex(quad, 2)),
   Connectivity{Edge, Vertex}(vertex(quad, 2), vertex(quad, 3)),
   Connectivity{Edge, Vertex}(vertex(quad, 3), vertex(quad, 4)),
   Connectivity{Edge, Vertex}(vertex(quad, 4), vertex(quad, 1)))
end

# todo: genralize
"Return canonical form of cell connectivity"
function canonicalize_connectivity(e::C) where C <: Connectivity{Edge, Vertex}
  e[1]<e[2] ? e : C(e[2], e[1])
end
