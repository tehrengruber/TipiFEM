import Base.show

function show(io::IO, ::Type{Polytope{id}}) where id
  write(io, "Polytope\"$(TipiFEM.PolytopalMesh.polytope_ids[id])\"")
end
