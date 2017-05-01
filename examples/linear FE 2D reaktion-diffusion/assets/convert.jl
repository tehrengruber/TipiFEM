using MAT
# Read in input file in .mat format
file = matopen("./assets/Square.mat")
varnames = names(file)
nodes = read(file, "Coordinates")
elements = read(file, "Elements")
# Write in .msh format
using ArraySlices
open("./assets/square.msh", "w") do f
    write(f, """\$MeshFormat
             2.2 0 8
             \$EndMeshFormat
             \$Nodes
             $(size(nodes, 1))
             """)
    for (i, node) in enumerate(rows(nodes))
        write(f, "$(i) $(join(node, " ")) 0.0\n")
    end
    write(f, "\$EndNodes\n")
    # elements
    write(f, "\$Elements\n$(size(elements, 1))\n")
    for (i, element) in enumerate(rows(elements))
        write(f, "$(i) 2 2 0 6 $(join(map(x->convert(Int, x), element), " "))\n")
    end
    write(f, "\$EndElements\n")
end
