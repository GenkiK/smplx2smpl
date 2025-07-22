def summary_closure(gt_vertices, var_dict, smpl_model):
    model_output = smpl_model(return_full_pose=True, **var_dict)
    est_vertices = model_output.vertices

    v2v = (est_vertices - gt_vertices).pow(2).sum(dim=-1).sqrt().mean()
    return {"Vertex-to-Vertex": v2v * 1000}
