using Iterators
import Base.show

function show{K <: Cell}(io::IO, ::MIME"text/plain", mesh_topology::MeshTopology{K}; simple=false, indent=0)
  mesh_dim=dim(K)
  let indent=isa(indent, Int) ? repeat(" ", indent) : indent,
      print=s->write(io, indent * s)
    simple || print("MeshTopology{$(K)}\n")
    print("→ | $(join(0:mesh_dim, "  "))\n")
    print("--|--------\n")
    for i=0:mesh_dim
      print("$(i) |")
      for j=0:mesh_dim
        write(io, " " * (ispopulated(mesh_topology, Dim{i}(), Dim{j}()) ? "1 " : "0 "))
      end
      print("\n")
    end
  end
end

import Base.show
function show(io::IO, ::MIME"text/plain", msh::Mesh)
    write(io,
          """TipiFEM.Mesh at $(pointer_from_objref(msh))
            world dim: $(world_dim(msh))
            mesh dimension: $(mesh_dim(msh))
            cell types: $(join(cell_types(msh), ", "))
          """)
    write(io, @sprintf "  %6s %6s %10s\n" "Codim" "Dim" "#cells")
    for d=convert(Int, mesh_dim(msh)):-1:0
        write(io, @sprintf "  %6i %6i %10i\n" (mesh_dim(msh)-d) d number_of_cells(msh, Dim{d}()))
    end
    write(io, "  topology: \n")
    show(io, MIME"text/plain"(), topology(msh), simple=true, indent=4)
end

function show(io::IO, ::MIME"text/html", msh::Mesh)
  html = """
  <table>
    <tr>
      <td colspan="2"><b>TipiFEM.Mesh</b> at $(pointer_from_objref(msh))</td>
    </tr>
    <tr>
      <td>world dimension</td>
      <td>$(world_dim(msh))</td>
    </tr>
    <tr>
      <td>mesh dimension</td>
      <td>$(mesh_dim(msh))</td>
    </tr>
    <tr>
      <td>cell types</td>
      <td>$(join(cell_types(msh), ", "))</td>
    </tr>
    <tr>
      <td colspan=2>
  """
  # cell count
  html *= let cell_count_html = """
      <table style="border:none; width: 100%">
        <thead>
          <tr>
            <td>Codim</td>
            <td>Dim</td>
            <td>#cells</td>
          </tr>
        </thead>
        <tbody>
      """
    for d=convert(Int, mesh_dim(msh)):-1:0
        cell_count_html *= """
          <tr>
            <td>$(mesh_dim(msh)-d)</td>
            <td>$(d)</td>
            <td>$(number_of_cells(msh, Dim{d}()))</td>
          </tr>
        """
    end
    cell_count_html *= "</tbody></table>"
  end
  # topology matrix
  html *= let s = BufferStream()
    show(s, MIME"text/plain"(), topology(msh))
    close(s)
    topology_mat_string = readstring(s)
    """
    <tr>
      <td>topology</td>
      <td><pre>$(topology_mat_string)</pre></td>
    </tr>
    """
  end
  html *= """
      </td>
    </tr>
  </table>"""
  write(io, html)
end
