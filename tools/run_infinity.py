import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from typing import List
import math
import time
import hashlib
import yaml
import argparse
import shutil
import re

import cv2
import numpy as np
import torch
torch._dynamo.config.cache_size_limit=64
import pandas as pd
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
from PIL import Image, ImageEnhance
import torch.nn.functional as F
from torch.cuda.amp import autocast

from infinity.models.infinity import Infinity
from infinity.models.basic import *
import PIL.Image as PImage
from torchvision.transforms.functional import to_tensor
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


def extract_key_val(text):
    pattern = r'<(.+?):(.+?)>'
    matches = re.findall(pattern, text)
    key_val = {}
    for match in matches:
        key_val[match[0]] = match[1].lstrip()
    return key_val

def get_word_token_mapping(text_tokenizer, prompt, target_words=None):
    """
    获取prompt中每个单词和指定短语对应的token范围。
    target_words: None 或 list[str]，如 ['cat'] 或 ['a cat']
    返回 dict: {word_or_phrase: (start, end)}
    """
    tokenized = text_tokenizer(prompt, return_offsets_mapping=True)
    word_ids = tokenized.word_ids()
    offsets = tokenized.offset_mapping
    words = prompt.split()

    # 单词到token范围
    word_to_tokens = {}
    current_word = None
    word_start = None
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id != current_word:
                if current_word is not None and current_word < len(words):
                    word_to_tokens[words[current_word]] = (word_start, i)
                current_word = word_id
                word_start = i
    if current_word is not None and current_word < len(words):
        word_to_tokens[words[current_word]] = (word_start, len(word_ids))

    # 如果没有指定 target_words，返回所有单词的映射
    if not target_words:
        return word_to_tokens

    # 支持短语：只查找 prompt 中实际出现的连续短语
    phrase_to_tokens = {}
    prompt_words = words
    for phrase in target_words:
        phrase_split = phrase.split()
        n = len(phrase_split)
        found = False
        for i in range(len(prompt_words) - n + 1):
            if prompt_words[i:i+n] == phrase_split:
                # 找到短语在 prompt 中的位置
                first_word = prompt_words[i]
                last_word = prompt_words[i+n-1]
                if first_word in word_to_tokens and last_word in word_to_tokens:
                    start_token = word_to_tokens[first_word][0]
                    end_token = word_to_tokens[last_word][1]
                    phrase_to_tokens[phrase] = (start_token, end_token)
                    found = True
                    break
        if not found:
            # 不在prompt中则跳过
            continue
    return phrase_to_tokens

def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False, embedding_offset_path=None):
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')
    if embedding_offset_path is not None:
        prompt = f"{prompt}, the language style of this prompt is"
    print(f'prompt={prompt}')
    captions = [prompt]
    tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()   # torch.Size([1, 512, 2048])

    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)    
    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)

    # 如果提供了embedding offset路径，则加载并添加到kv_compact上
    if embedding_offset_path is not None:
        print(f'Loading embedding offset from {embedding_offset_path}')
        embedding_offset = torch.load(embedding_offset_path, map_location='cuda')
        concat_length = 20
        offset_to_concat = embedding_offset[0, :concat_length, :]
        kv_compact = torch.cat([kv_compact, offset_to_concat], dim=0)
        # 更新相关参数
        lens = [l + concat_length for l in lens]  # 每个序列的长度+20
        cu_seqlens_k = torch.cat([cu_seqlens_k[:-1], cu_seqlens_k[-1:] + concat_length], dim=0)
        Ltext = Ltext + concat_length  # 最大序列长度+20

    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple

def aug_with_positive_prompt(prompt):
    for key in ['man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'person', 'human', 'adult', 'teenager', 'employee', 
                'employer', 'worker', 'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather', 'son', 'daughter']:
        if key in prompt:
            prompt = prompt + '. very smooth faces, good looking faces, face to the camera, perfect facial features'
            break
    return prompt

def enhance_image(image):
    for t in range(1):
        contrast_image = image.copy()
        contrast_enhancer = ImageEnhance.Contrast(contrast_image)
        contrast_image = contrast_enhancer.enhance(1.05)  # 增强对比度
        color_image = contrast_image.copy()
        color_enhancer = ImageEnhance.Color(color_image)
        color_image = color_enhancer.enhance(1.05)  # 增强饱和度
    return color_image

def gen_one_img(
    infinity_test, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    infer_function=0,  #设置启动inference的什么功能
    infer_root_dir=None,  # 添加根目录参数
    infer_sub_dir=None,  # 添加子目录参数
    use_concat_embedding=0,
    use_embedding_iter=0,  # 添加迭代次数参数 
    target_words=None,  # 添加目标单词列表
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)

    # 获取word到token的映射
    word_token_mapping = get_word_token_mapping(text_tokenizer, prompt, target_words)
    
    print(f"Target words: {target_words}")
    print(f"Word to token mapping: {word_token_mapping}")

    file_part = f"trained_language_style_embedding/embedding_it{use_embedding_iter-1}.pth"
    full_path = os.path.join(infer_root_dir, infer_sub_dir, file_part)

    if use_concat_embedding:
        text_cond_tuple = encode_prompt(
        text_tokenizer, 
            text_encoder, 
            prompt, 
            enable_positive_prompt=enable_positive_prompt,
            embedding_offset_path=full_path
        )
    else:
        text_cond_tuple = encode_prompt(
            text_tokenizer, 
            text_encoder, 
            prompt, 
            enable_positive_prompt=enable_positive_prompt,
        )

    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None
    print(f'cfg: {cfg_list}, tau: {tau_list}')

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        _, _, img_list = infinity_test.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            infer_function=infer_function,  # 传递infer_function参数，选择infer需要实现的功能
            infer_root_dir=infer_root_dir,  # 传递根目录参数
            infer_sub_dir=infer_sub_dir,  # 传递子目录参数
            word_token_mapping=word_token_mapping,  # 传递word到token的映射
            target_words=target_words,  # 传递目标单词列表
        )
    print(f"cost: {time.time() - sstt}, infinity cost={time.time() - stt}")
    img = img_list[0]
    return img

def get_prompt_id(prompt):
    md5 = hashlib.md5()
    md5.update(prompt.encode('utf-8'))
    prompt_id = md5.hexdigest()
    return prompt_id

def save_slim_model(infinity_model_path, save_file=None, device='cpu', key='gpt_fsdp'):
    print('[Save slim model]')
    full_ckpt = torch.load(infinity_model_path, map_location=device)
    infinity_slim = full_ckpt['trainer'][key]
    # ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
    if not save_file:
        save_file = osp.splitext(infinity_model_path)[0] + '-slim.pth'
    print(f'Save to {save_file}')
    torch.save(infinity_slim, save_file)
    print('[Save slim model] done')
    return save_file

def load_tokenizer(t5_path =''):
    print(f'[Loading tokenizer and text encoder]')
    text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
    text_tokenizer.model_max_length = 512
    text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16)
    text_encoder.to('cuda')
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return text_tokenizer, text_encoder

def load_infinity(
    rope2d_each_sa_layer, 
    rope2d_normalized_by_hw, 
    use_scale_schedule_embedding, 
    pn, 
    use_bit_label, 
    add_lvl_embeding_only_first_block, 
    model_path='', 
    scale_schedule=None, 
    vae=None, 
    device='cuda', 
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
    checkpoint_type='torch',
    use_lora=False,
    lora_r=4,
    lora_alpha=32,
    lora_dropout=0.1,
    use_lora_iter=50,
    infer_root_dir=None,
    infer_sub_dir=None,
):
    print(f'[Loading Infinity]')
    text_maxlen = 512
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity_test: Infinity = Infinity(
            vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
            shared_aln=True, raw_scale_schedule=scale_schedule,
            checkpointing='full-block',
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to(device=device)
        print(f'[you selected Infinity with {model_kwargs=}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')

        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)

        infinity_test.cuda()
        torch.cuda.empty_cache()

        print(f'[Load Infinity weights]')
        if checkpoint_type == 'torch':
            state_dict = torch.load(model_path, map_location=device)
            print(infinity_test.load_state_dict(state_dict))
        elif checkpoint_type == 'torch_shard':
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(infinity_test, model_path, strict=False)
        infinity_test.rng = torch.Generator(device=device)

        if use_lora:
            print(f'[Using LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, iter={use_lora_iter}]')
            #replace ffn linear with lora
            infinity_test.replace_ffn_linear_with_lora(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )

            # infinity_test.replace_attention_linear_with_lora(
            #     r=lora_r,
            #     lora_alpha=lora_alpha,
            #     lora_dropout=lora_dropout
            # )
            
            lora_weights_path = os.path.join(infer_root_dir, infer_sub_dir, f'lora_weight/lora_weight_{use_lora_iter}iter.pt')
            lora_weights = torch.load(lora_weights_path, map_location=torch.device('cpu'))

            # 3. 过滤并仅加载LoRA相关参数
            model_state_dict = infinity_test.state_dict()

            # 更新模型状态字典，仅保留LoRA权重
            for name, param in lora_weights.items():
                if name in model_state_dict:
                    if 'lora_A' in name:
                        # 重塑 lora_A 为 [lora_r, -1]
                        model_state_dict[name].copy_(param.reshape(lora_r, -1))
                    elif 'lora_B' in name:
                        # 重塑 lora_B 为 [-1, lora_r]
                        model_state_dict[name].copy_(param.reshape(-1, lora_r))
                else:
                    print(f"Warning: LoRA weight '{name}' not found in model. Skipping.")

        return infinity_test

def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)

def joint_vi_vae_encode_decode(vae, image_path, scale_schedule, device, tgt_h, tgt_w):
    pil_image = Image.open(image_path).convert('RGB')
    inp = transform(pil_image, tgt_h, tgt_w)
    inp = inp.unsqueeze(0).to(device)
    scale_schedule = [(item[0], item[1], item[2]) for item in scale_schedule]
    t1 = time.time()
    h, z, _, all_bit_indices, _, infinity_input = vae.encode(inp, scale_schedule=scale_schedule)
    t2 = time.time()
    recons_img = vae.decode(z)[0]
    if len(recons_img.shape) == 4:
        recons_img = recons_img.squeeze(1)
    print(f'recons: z.shape: {z.shape}, recons_img shape: {recons_img.shape}')
    t3 = time.time()
    print(f'vae encode takes {t2-t1:.2f}s, decode takes {t3-t2:.2f}s')
    recons_img = (recons_img + 1) / 2
    recons_img = recons_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    gt_img = (inp[0] + 1) / 2
    gt_img = gt_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    print(recons_img.shape, gt_img.shape)
    return gt_img, recons_img, all_bit_indices

def load_visual_tokenizer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vae
    if args.vae_type in [14,16,18,20,24,32,64]:
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae = vae_model(args.vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae

def load_transformer(vae, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    if args.checkpoint_type == 'torch': 
        # copy large model to local; save slim to local; and copy slim to nas; load local slim model
        if osp.exists(args.cache_dir):
            local_model_path = osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_'))
        else:
            local_model_path = model_path
        if args.enable_model_cache:
            slim_model_path = model_path.replace('ar-', 'slim-')
            local_slim_model_path = local_model_path.replace('ar-', 'slim-')
            os.makedirs(osp.dirname(local_slim_model_path), exist_ok=True)
            print(f'model_path: {model_path}, slim_model_path: {slim_model_path}')
            print(f'local_model_path: {local_model_path}, local_slim_model_path: {local_slim_model_path}')
            if not osp.exists(local_slim_model_path):
                if osp.exists(slim_model_path):
                    print(f'copy {slim_model_path} to {local_slim_model_path}')
                    shutil.copyfile(slim_model_path, local_slim_model_path)
                else:
                    if not osp.exists(local_model_path):
                        print(f'copy {model_path} to {local_model_path}')
                        shutil.copyfile(model_path, local_model_path)
                    save_slim_model(local_model_path, save_file=local_slim_model_path, device=device)
                    print(f'copy {local_slim_model_path} to {slim_model_path}')
                    if not osp.exists(slim_model_path):
                        shutil.copyfile(local_slim_model_path, slim_model_path)
                        os.remove(local_model_path)
                        os.remove(model_path)
            slim_model_path = local_slim_model_path
        else:
            slim_model_path = model_path
        print(f'load checkpoint from {slim_model_path}')
    elif args.checkpoint_type == 'torch_shard':
        slim_model_path = model_path

    if args.model_type == 'infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8) # 2b model
    elif args.model_type == 'infinity_8b':
        kwargs_model = dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    elif args.model_type == 'infinity_layer12':
        kwargs_model = dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer16':
        kwargs_model = dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer24':
        kwargs_model = dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer32':
        kwargs_model = dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer40':
        kwargs_model = dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer48':
        kwargs_model = dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)

    
    infinity = load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer, 
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label, 
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block, 
        model_path=slim_model_path, 
        scale_schedule=None, 
        vae=vae, 
        device=device, 
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
        checkpoint_type=args.checkpoint_type,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_lora_iter=args.use_lora_iter,
        infer_root_dir=args.infer_root_dir,
        infer_sub_dir=args.infer_sub_dir,
    )
    return infinity

def add_common_arguments(parser):
    parser.add_argument('--cfg', type=str, default='3')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--pn', type=str, required=True, choices=['0.06M', '0.25M', '1M'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=1)
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=0, choices=[0,1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0,1])
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_encoder_ckpt', type=str, default='')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--enable_model_cache', type=int, default=0, choices=[0,1])
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])
    parser.add_argument('--infer_function', type=int, default=0, choices=[0,1,2,3,4], help='0: original_autoregressive_func, 1: get_original_branch_self_attn_kv, 2: remove_object_func, 3: add_object_func, 4: get_edit_branch_self_attn_kv')
    # Add text embedding related arguments
    parser.add_argument('--train_textembedding', type=int, default=0, choices=[0,1], help='Whether to train text embedding')
    parser.add_argument('--train_textembedding_iter', type=int, default=10, help='Text embedding iteration number')
    # Add LoRA related arguments
    parser.add_argument('--use_lora', type=int, default=0, choices=[0,1], help='Whether to use LoRA when inference')
    parser.add_argument('--train_lora', type=int, default=0, choices=[0,1], help='Whether to train LoRA when training')
    parser.add_argument('--lora_r', type=int, default=4, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate')
    parser.add_argument('--use_lora_iter', type=int, default=50, help='LoRA iteration number')
    # Add inference directory arguments
    parser.add_argument('--infer_root_dir', type=str, default='/data1/chenyuxin/code/Infinity_clone/PIE-Benchmark-edit-v1/annotation_images/3_delete_object_80/', help='Root directory for inference')
    parser.add_argument('--infer_sub_dir', type=str, default='1_artificial/1_animal/311000000000/', help='Sub directory for inference')
    parser.add_argument('--use_concat_embedding', type=int, default=0, choices=[0,1], help='Whether to use concat language style embedding')
    parser.add_argument('--use_embedding_iter', type=int, default=0, help='Iteration number for concat language style embedding')
    parser.add_argument('--target_words', type=str, default=None, help='Target words for cross attention analysis (comma-separated)')

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--prompt', type=str, default='a dog')
    parser.add_argument('--save_file', type=str, default='./tmp.jpg')
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    # parse target_words
    if args.target_words:
        args.target_words = [word.strip() for word in args.target_words.split(',')]
        print(f"Target words: {args.target_words}")
    else:
        args.target_words = None
    
    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path =args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)
    
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [ (1, h, w) for (_, h, w) in scale_schedule]

    # 修改这部分路径处理代码，为了之后可以concat
    if args.infer_sub_dir.startswith('/'):
        # 如果是绝对路径，移除开头的斜杠
        infer_sub_dir = args.infer_sub_dir[1:] if args.infer_sub_dir.startswith('/') else args.infer_sub_dir
    else:
        infer_sub_dir = args.infer_sub_dir

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            generated_image = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                args.prompt,
                g_seed=args.seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
                infer_function=args.infer_function,
                infer_root_dir=args.infer_root_dir,
                infer_sub_dir=infer_sub_dir,
                use_concat_embedding=args.use_concat_embedding,
                use_embedding_iter=args.use_embedding_iter,
                target_words=args.target_words,  # 使用命令行参数指定的目标单词
            )
    os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
    cv2.imwrite(args.save_file, generated_image.cpu().numpy())
    print(f'Save to {osp.abspath(args.save_file)}')
