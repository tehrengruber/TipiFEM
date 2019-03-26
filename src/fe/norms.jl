using TipiFEM.Quadrature

"""
Compute the L2 norm of the difference of μ with u
"""
function l2_norm(fespace, μ, u)
  quadrule_list = QuadruleList(Quadrule{Polytope"3-node triangle", 25, Float64},
                               Quadrule{Polytope"4-node quadrangle", 36, Float64})
  let dofh = dofh(fespace), basis=basis(fespace), mesh=mesh(fespace)
    result = 0.
    foreach(decompose(graph(geometry(mesh)))) do mesh_geo
      for (cid, geo) in mesh_geo
        res = 0.
        dofs = dofh[cid]
        ŵs, x̂s, ŝ = quadrule_list[cell_type(cid)]
        for (ŵ, x̂) in zip(ŵs, x̂s)
          let lsfs = local_shape_functions(basis, cell_type(cid)(), x̂),
              det_DΦ = integration_element(geo, x̂),
              x = local_to_global(geo, x̂)
            val = 0.
            for i in 1:length(lsfs)
              μ_i = μ[dofs[i]]*lsfs[i]
              val += μ_i
            end
            res += det_DΦ * ŵ * (val-u(x)).^2
          end
        end
        result+=res*ŝ
      end
    end
    result
  end
end
