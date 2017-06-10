using Base.@pure

"Singleton type representing a convex polytope"
immutable Polytope{id} <: Cell end

"""
The ids and names are borrowed from the gmsh file format
see Gmsh reference http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format
"""
const polytope_ids = Dict(
	1  => "2-node line",
	2  => "3-node triangle",
	3  => "4-node quadrangle",
	4  => "4-node tetrahedron",
	5  => "8-node hexahedron",
	6  => "6-node prism",
	7  => "5-node pyramid",
	#8  => "3-node second order line",
	#9  => "6-node second order triangle",
	#10 => "9-node second order quadrangle",
	#11 => "10-node second order tetrahedron",
	#12 => "27-node second order hexahedron",
	#13 => "18-node second order prism",
	#14 => "14-node second order pyramid",
	15 => "1-node point",
	#16 => "8-node second order quadrangle",
	#17 => "20-node second order hexahedron",
	#18 => "15-node second order prism",
	#19 => "13-node second order pyramid",
	#20 => "9-node third order incomplete triangle",
	#21 => "10-node third order triangle",
	#22 => "12-node fourth order incomplete triangle",
	#23 => "15-node fourth order triangle",
	#24 => "15-node fifth order incomplete triangle",
	#25 => "21-node fifth order complete triangle",
	#26 => "4-node third order edge",
	#27 => "5-node fourth order edge",
	#28 => "6-node fifth order edge",
	#29 => "20-node third order tetrahedron",
	#30 => "35-node fourth order tetrahedron",
	#31 => "56-node fifth order tetrahedron",
	#92 => "64-node third order hexahedron",
	#93 => "125-node fourth order hexahedron"
)

"Polytopes supported by PolytopalKomplex"
const polytopes = collect(Polytope{id} for id in keys(polytope_ids))

"Map name of a polytope to its id"
const polytope_ids_transpose = map((p) -> p[2] => p[1], polytope_ids)

"Map polytopes to their number of vertices"
const polytope_vertex_count = Dict(collect((id, parse(Int, match(r"([0-9]+)", name)[1])) for (id, name) in polytope_ids))

"""
Expand an expression like `Polytope"1-node point"` into the corresponding
datatype `Polytope{15}`
"""
macro Polytope_str(s)
    :(Polytope{$(polytope_ids_transpose[s])})
end

"""
Expand an expression like `Id"1-node point"` into the corresponding
datatype `Id{Polytope"1-node point"}`
"""
macro Id_str(s) :(Id{$(macroexpand(:(@Polytope_str($(s)))))}) end

Polytope(s::String) = Polytope{polytope_ids_transpose[s]}()

#function Polytope{id}(indices::Vararg{Id"1-node point"})
#	Connectivity{Polytope{id},
#							 Polytope"1-node point",
#							 vertex_count(Polytope{id})}(indices...)
#end

for id in keys(polytope_ids)
	@eval begin
		"Number of vertices of the given convex polytope"
		@Base.pure vertex_count(::Polytope{$(id)}) = $(polytope_vertex_count[id])
		@Base.pure vertex_count(::Type{Polytope{$(id)}}) = $(polytope_vertex_count[id])

		@Base.pure face_count(::Polytope{$(id)}, ::Type{Polytope"1-node point"}) = $(polytope_vertex_count[id])
		@Base.pure face_count(::Type{Polytope{$(id)}}, ::Type{Polytope"1-node point"}) = $(polytope_vertex_count[id])
	end
end

################################################################################



################################################################################

parse_connectivity_string(s::String) = let sep_pos=search(s, '→')
    typeof(Polytope(strip(s[1:sep_pos-1]))),
    typeof(Polytope(strip(s[sep_pos+3:end])))
end

macro Connectivity_str(s)
    CT, FT = parse_connectivity_string(s)
    :(Connectivity{$(CT), $(FT)})
end

const Vertex = Polytope"1-node point"
const Edge = Polytope"2-node line"

#import Base.getindex
#using ComputedFieldTypes
#Base.@propagate_inbounds getindex(v::fulltype(Connectivity{Polytope"4-node quadrangle", Polytope"1-node point"}), i::Integer) = v.data[i]
