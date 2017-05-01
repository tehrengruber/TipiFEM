import Base.show

function show{id}(io::IO, ::Type{Polytope{id}})
  write(io, "Polytope\"$(TipiFEM.PolytopalMesh.polytope_ids[id])\"")
end
