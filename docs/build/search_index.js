var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#TipiFEM-Framework-for-the-Finite-Element-Method-1",
    "page": "Home",
    "title": "TipiFEM - Framework for the Finite Element Method",
    "category": "section",
    "text": "This is the documentation to TipiFEM, a framework for the implementation of Finite Element Methods in julia. The purpose of this documentation is to give the required mathematical background on which TipiFEM builds on, explain the software design of TipiFEM and how it may be used.Since there are many different choices to make when implementing a specific FEM the package is separated into different modules.The framework is seperated into different modulesSupport for different element types\n3-noded triangular elements\n4-noded quadrilateral elements"
},

{
    "location": "index.html#TipiFEM.Meshes.Mesh",
    "page": "Home",
    "title": "TipiFEM.Meshes.Mesh",
    "category": "Type",
    "text": "Construct a generic mesh containing K cells.\n\nMesh(Polygon\"3-node triangle\")\nMesh(Union{Polygon\"3-node triangle\", Polygon\"4-node quadrangle\"})\n\n\n\n"
},

{
    "location": "index.html#TipiFEM.Meshes.world_dim",
    "page": "Home",
    "title": "TipiFEM.Meshes.world_dim",
    "category": "Function",
    "text": "dimension of the ambient space\n\n\n\n"
},

{
    "location": "index.html#TipiFEM.Meshes.cell_type",
    "page": "Home",
    "title": "TipiFEM.Meshes.cell_type",
    "category": "Function",
    "text": "type of cells with\n\n\n\n"
},

{
    "location": "index.html#TipiFEM.Meshes.mesh_dim",
    "page": "Home",
    "title": "TipiFEM.Meshes.mesh_dim",
    "category": "Function",
    "text": "dimension of codimension zero cells\n\n\n\n"
},

{
    "location": "index.html#TipiFEM.Meshes.real_type",
    "page": "Home",
    "title": "TipiFEM.Meshes.real_type",
    "category": "Function",
    "text": "type used for calculations with real numbers\n\n\n\n"
},

{
    "location": "index.html#Mesh-1",
    "page": "Home",
    "title": "Mesh",
    "category": "section",
    "text": "We define a mesh cal M as subdivision of a bounded set Omega subset mathbbR^d into open cells of dimension  d, where a cell is in general a topological space homeomorph to an open ball. The mesh is then used to represent the domain on which the problem we are trying to solve with our FEM solver is posed on.The mesh dimension or simply dimension of a mesh is the largest dimension of one of its cells. The dimension of the ambient space d of the mesh is called the world dimension. We will generally use the notation K or K_i to talk about cells of codimension zero.High level viewA mesh in TipiFEM is stored as an object of type Mesh consisting of multiple MeshFunctions, which store the vertex coordinates, cell topology and additional cell attributes.RestrictionsIn TipiFEM we further restrict Omega to be a manifold surface and the cells to be polytopes or entities with the same combinatorial structure as polytopes (e.g. curvilinear elements). The first restriction is due to the algorithm that is used to wire up the incidence relation between different cells and might be lifted by a different implementation there. The mesh data structure itself may easily store non-manifold meshes. The restriction on the type of cells is only made since I wasn't aware of any problem that may be solved using finite elements where the cells have a different combinatorial structure then polytopes.FeaturesSupport for different element types\n3-noded triangular elements\n4-noded quadrilateral elements\nGeneric mesh datastructureNow that we have defined what a mesh is we proceed with its implementation. A minimal representation of a polytopal-like mesh consists of a set of vertices cal V, where each vertex has a unique identifier and coordinate x in Omega, and a set of cells cal K, where each cell has again a unique identifier and set of vertices.TipiFEM.Meshes.Mesh\nTipiFEM.Meshes.world_dim\nTipiFEM.Meshes.cell_type\nTipiFEM.Meshes.mesh_dim\nTipiFEM.Meshes.real_type"
},

{
    "location": "index.html#MeshInterface-1",
    "page": "Home",
    "title": "MeshInterface",
    "category": "section",
    "text": "immutable Edge <: Cell end dim(::Type{Edge}) = 1 face_count(::Type{Edge}, ::Type{Vertex}) = 2 facet(::Type{Edge}) = Vertex volume(geo::Geometry{Edge}) = point(geo, 2)[1] - point(geo, 1)[1]Many aspects of a mesh implementation are The topology and geometry of the mesh TipiFEM contains a generic mesh implementation that may be used for various__Simple1DMesh____PolytopalMesh__"
},

{
    "location": "index.html#MeshFunction-1",
    "page": "Home",
    "title": "MeshFunction",
    "category": "section",
    "text": "The basic building block to store both geometrical and combinatorial information of a mesh is what we call a mesh function, which is just a discrete function that maps cell-indices of cells with a fixed dimension to arbitrary objects.TipiFEM.Meshes.AbstractMeshFunction{II, VI}Lets start with some examples of how mesh functions may be used to store the topology of a mesh. We begin with a simple unstructed two dimensional mesh as of Fig #X by constructing a mesh function that maps indices of triangles to vertex indices.using TipiFEM, TipiFEM.Meshes, TipiFEM.PolytopalMesh\nmf = MeshFunction(Polytope\"3-node triangle\", NTuple{3, Index\"1-node point\"})\npush!(mf, (1, 4, 2))\npush!(mf, (1, 3, 4))\npush!(mf, (1, 3, 4))The value table mf is then:i f(i)\nIndex\"3-node triangle\"(1) (Index\"1-node point\"(1), Index\"1-node point\"(4), Index\"1-node point\"(2))\nIndex\"3-node triangle\"(2) (Index\"1-node point\"(1), Index\"1-node point\"(3), Index\"1-node point\"(4))Note how automatically two indices for the triangles are created and how the tuple of integers is converted into a tuple of indices. We can now look at the internal storage representation of the mesh function.domain(mf)\nimage(mf)We see that the indices in this case are stored as a leightweight UnitRange object of triangle indices, where a UnitRange object stores only the starting and end value of a range eliminated the need to store all indices or to store indices implicitly by their index. The image of the mesh function however is stored as a regular Array of tuples.Now that we have a mesh function that contains some values we can further explore some aspects of the MeshFunction interface. The mesh function may be evaluated at any index by calling the [] operator.mf[Index\"3-node triangle\"(1)]In case we know the integer index of the value inside the values array we can also evaluate the function using that index instead of the triangle index, which is not an integer by itself, eliminating the computation of the integer index from the triangle index.mf[1]Note that in this case the integer index and the triangle index are equal, which will not be the case if the domain is not a UnitRange beginning at one (see example #X).Iteration over all values works just like with a regular Array:for connectivity in mf\n  println(connectivity)\nendHowever in some cases the indices are also required for which the graph function is useful:for (index, connectivity) in graph(mf)\n  println(\"index: \", index, \", connectivity: \", connectivity)\nendJust like iteration over a mesh function works as if it was a regular array, many other functions that work on regular arrays also work with mesh functions (e.g. map, reduce etc.). We might for example want obtain all edges of the triangles stored in mf. Assuming a fixed node numbering convention, where the vertices of each triangle are stored in counter clockwise order, we can get all edges like this:# construct a mesh function that maps triangles to the connectivity of all of its edges\ntriangle_edges = map(mf) do connectivity\n  ((connectivity[1], connectivity[2]),\n   (connectivity[2], connectivity[3]),\n   (connectivity[3], connectivity[1]))\nend\n# remove duplicate edges\nedges = MeshFunction(Edge, NTuple{2, Index\"1-node point\"})\nfor edges in triangle_edges\n  for edge in triangle_edges\n    if edge[2]<edge[1]\n      push!(edges, edge)\n    end\n  end\nendNow we proceed with a more complex examplemap indices of triangles and quadrangles to vertex indices. For this purpose we can either construct two mesh functions, one for the triangles and one for the quadrangles, and union them or create a single mesh function that stores both triangles and quadrangles.mf1 = MeshFunction(Polytope\"3-node triangle\", Array{Vertex, 1})()\nmf2 = MeshFunction(Polytope\"3-node quadrangle\", Array{Vertex, 1})()\npush!(mf1, [1, 4, 2])\npush!(mf2, [1, 3, 4])\nmf = mf1 ∪ mf2The construction of a mesh cell is straightforward by callingThe mesh is represented as a set of mesh functions, which we define as a discrete function that maps cell-indices of cells with a fixed dimension to arbitrary objects.The central data-type used by the mesh is a MeshFunction. It is defined as a discrete function that maps cell-indices of cells with a fixed dimension to arbitrary objects. MeshFunctions are used to store the connectivity, material parameters or user defined attributes of mesh cells. Together a set of mesh functions may then define all properties of the cells in the mesh and give them an identity through their index. Currently there exist two different concrete types of MeshFunctions.The discreteness of a MeshFunction is expressed byEssentially a MeshFunction is just an interface around a pair of equally sized AbstractArrays storing cell indices and values. Since the meaning of a MeshFunction is only clear in the context it is being used the__Scalar discrete function__"
},

{
    "location": "index.html#A-simple-mesh-1",
    "page": "Home",
    "title": "A simple mesh",
    "category": "section",
    "text": "__Connectivity__trias = MeshFunction(Polytope\"3-node triangle\", SVector{3, Index\"1-node point\"})\npush!(trias, [v1, v4, v2])\npush!(trias, [v1, v3, v4])__Geometry__nodes = MeshFunction(Polytope\"1-node point\", SVector{2, Float64})\nv1 = push!(nodes, [0, 0])\nv2 = push!(nodes, [0, 1])\nv3 = push!(nodes, [1, 0])\nv4 = push!(nodes, [0.5, 0.5])In combination with the connectivity the vertex coordinates may be used to calculate the area of all trianglesTipiFEM.Meshes.domain\nTipiFEM.Meshes.image\nTipiFEM.Meshes.graph\nTipiFEM.Meshes.idxtype\nTipiFEM.Meshes.cell_type\nTipiFEM.Meshes.eltype\nTipiFEM.Meshes.length\nTipiFEM.Meshes.start\nTipiFEM.Meshes.done\nTipiFEM.Meshes.next\nTipiFEM.Meshes.getindex\nTipiFEM.Meshes.mapThe MeshFunction type has two fields, one for the cell indicesThe actual storage layout__Example__ Let's begin with an discrete scalar function on a set of triangles. Such a function may for example assign each triangle its volume.# construct a mesh function mapping triangles to Float64 (Real) values\nmf=MeshFunction(Polytope\"3-node triangle\", Float64)Adding values is as easy as:push!(mf, Index\"3-node triangle\"(1), 1)\npush!(mf, Index\"3-node triangle\"(1), 2)Now just like with a regular array sum(mf)sum(mf)MeshFunction(Union{Polytope\"4-node quadrangle\", Polytope\"3-node triangle\"}, Float64)MeshTopologyThe topology of the mesh is represented by incidence relations stored as a (d+1) times (d+1) Matrix 𝕀 of Connectivity-valued MeshFunctions, where 𝕀(d_1 d_2) stores the incident d_2 dimensional cells of d_1 dimensional cells."
},

{
    "location": "index.html#MeshCell-1",
    "page": "Home",
    "title": "MeshCell",
    "category": "section",
    "text": "Since the cells of a mesh are stored implicitly through MeshFunctions their data-types also do not store any information. A concrete MeshCell (e.g. a Polytope) is therefore implemented as a singleton type in julia, whose only purpose is to allow distinguishing different types of cells. Intrinsic properties of mesh cells may then be defined by simple functions taking a concrete MeshCell as an argument. The dimension of a 3-node triangle for example is defined like this:dim(::Polytope\"3-node triangle\") = 2"
},

{
    "location": "index.html#Connectivity-1",
    "page": "Home",
    "title": "Connectivity",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#Geometry-1",
    "page": "Home",
    "title": "Geometry",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#Misc-1",
    "page": "Home",
    "title": "Misc",
    "category": "section",
    "text": "Dim, CodimThe Dim and Codim types are used to represent dimension and codimension information. Since in most situations the dimension and the codimension is already known at compile time these types are implemented as singleton types that store the value of the (co)dimension as a parametric type (note that the value is an integer and not a type). This allows us to dispatch on the (co)dimension, which is especially handy in cases, where the knowledge of the (co)dimension at compile time allows generation of efficient code.Connectivity\"3-node triangle → 1-node point\"Mandatory information about"
},

{
    "location": "index.html#DOF-Handler-1",
    "page": "Home",
    "title": "DOF Handler",
    "category": "section",
    "text": "The dof-handler maps dofs to unique integer indices. A simple dof handler that works for a nodal basis may be implemented in two steps. The first step, an initialization procedure, has the be run only once. Here we assign each cell type a disjoint integer range as large as the number of cells of this type in the mesh. Now everyimplemented by first assigning disjoint integer ranges to each cell type and using the starting point of each interval as an offsetthe number of cells per type andsum_K cells_per_type(K)"
},

{
    "location": "index.html#Quadrature-1",
    "page": "Home",
    "title": "Quadrature",
    "category": "section",
    "text": "The Quadrature module provides generic code for the approximation of integrals on a mesh by means of a weighted sum of point values.int_K f(mathbbx) mathrmdK approx sum_j=1^n w_j^n f(c_j^n)Evaluation of the approximation is by means of calling integrate(f, Kis) with f the integrand f in procedural form and Kis a set of cells given as a MeshFunction of cell Geometry objects.The concrete mesh implementation is then only obligated to define quadrature weights and nodes before the integrateThe domain in form of a Mesh object and integrand in procedural form may then be given to integrantThe concrete mesh implmentation defines quadrature weights and nodes get defined byFor the approximation of integrals on functions given in procedural form"
},

{
    "location": "index.html#Matrix-Assembler-1",
    "page": "Home",
    "title": "Matrix Assembler",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#Vector-Assembler-1",
    "page": "Home",
    "title": "Vector Assembler",
    "category": "section",
    "text": "(Image: Image of Yaktocat)"
},

]}
