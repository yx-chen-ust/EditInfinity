def gen_one_img(model, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0, si=None, block_idx=None, module_idx=None, save_self_attn_kv=False, replace_self_attn_kv=False, infer_root_dir=None, infer_sub_dir=None):
    with torch.no_grad():
        x = model(x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind, si, block_idx, module_idx, save_self_attn_kv, replace_self_attn_kv, infer_root_dir, infer_sub_dir)
    return x 