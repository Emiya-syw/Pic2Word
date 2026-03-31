# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from functools import partial
from torch.cuda.amp import autocast
import torch.distributed as dist
from tqdm import tqdm
from torchvision.utils import save_image
import sys
import pdb
import logging
import torch.nn.functional as F
from third_party.open_clip.clip import tokenize, _transform
import pickle
from third_party.open_clip import clip as open_clip
from utils import is_master

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def tokenize_to_device(texts, args):
    """
    Support:
      1) raw texts: list[str] / tuple[str] / str
      2) pre-tokenized tensor: Tensor[B, L]
    """
    if isinstance(texts, torch.Tensor):
        tokens = texts
        if args.gpu is not None:
            tokens = tokens.cuda(args.gpu, non_blocking=True)
        return tokens.long()

    if isinstance(texts, str):
        texts = [texts]

    if isinstance(texts, (list, tuple)):
        if len(texts) == 0:
            raise ValueError("Empty text batch.")
        first = texts[0]
        if isinstance(first, str):
            tokens = tokenize(list(texts))
            if args.gpu is not None:
                tokens = tokens.cuda(args.gpu, non_blocking=True)
            return tokens.long()
        if isinstance(first, torch.Tensor):
            tokens = torch.stack(list(texts), dim=0)
            if args.gpu is not None:
                tokens = tokens.cuda(args.gpu, non_blocking=True)
            return tokens.long()

    raise TypeError(f"Unsupported texts type: {type(texts)}")


def encode_text_batch(model, texts, args):
    tokens = tokenize_to_device(texts, args)
    text_features = model.encode_text(tokens)
    return text_features


def apply_global_start_noise(features, args):
    noise_std = getattr(args, "global_start_noise_std", 0.0)
    if noise_std <= 0:
        return features
    return features + torch.randn_like(features) * noise_std


def get_text_features(model, token_features, args):
    """
    image-derived token features -> CLIP text encoder
    same logic as training
    """
    text = tokenize("a photo of")
    if args.gpu is not None:
        text = text.cuda(args.gpu, non_blocking=True)

    text = text.view(1, -1)
    text = text.repeat(token_features.size(0), 1).long()
    text_features = model.encode_text_img(text, token_features)
    return text_features


def encode_image_via_img2text(model, img2text, images, args):
    """
    ref image -> image encoder -> img2text -> text encoder
    """
    image_features = model.encode_image(images)
    token_features = img2text(image_features)
    text_features = get_text_features(model, token_features, args)
    return text_features


def _expand_pic2word_marker_slots(text_tokens, marker_token_id, num_slots):
    if num_slots <= 1:
        return text_tokens

    context_length = text_tokens.size(1)
    eot_token_id = tokenize([""])[0][1].item()
    expanded_rows = []
    for row in text_tokens:
        seq = row.tolist()
        valid_len = 0
        for token_id in seq:
            if token_id == 0:
                break
            valid_len += 1
        valid_seq = seq[:valid_len]
        try:
            marker_idx = valid_seq.index(marker_token_id)
        except ValueError:
            expanded_rows.append(row)
            continue

        new_seq = valid_seq[:marker_idx] + [marker_token_id] * num_slots + valid_seq[marker_idx + 1 :]
        if len(new_seq) > context_length:
            new_seq = new_seq[:context_length]
            if eot_token_id not in new_seq:
                new_seq[-1] = eot_token_id

        padded = new_seq + [0] * (context_length - len(new_seq))
        expanded_rows.append(row.new_tensor(padded))

    return torch.stack(expanded_rows, dim=0)


def _normalize_feature(x, eps=1e-6):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def encode_pic2word_composed_feature(model, img2text, images, texts, args):
    text_tokens = tokenize_to_device(texts, args)
    split_text = getattr(args, "global_flow_pic2word_marker", "*")
    split_token_id = tokenize([split_text])[0][1].item()
    pic2word_topk_text = max(0, int(getattr(args, "global_flow_pic2word_topk_text", 0)))
    num_pic2word_tokens = max(1, pic2word_topk_text)
    text_tokens = _expand_pic2word_marker_slots(text_tokens, split_token_id, num_pic2word_tokens)

    if not torch.all((text_tokens == split_token_id).any(dim=1)):
        raise ValueError(
            "global_flow_compose_method='pic2word' requires every text prompt to "
            f"contain the marker token {split_text!r}."
        )

    image_features = model.encode_image(images)
    pseudo_word_embedding = img2text(image_features)
    if pic2word_topk_text <= 0:
        query_image_tokens = pseudo_word_embedding
    else:
        token_bank = model.token_embedding.weight.type(model.dtype)
        pseudo_norm = F.normalize(pseudo_word_embedding.type(model.dtype), dim=-1)
        token_bank_norm = F.normalize(token_bank, dim=-1)
        nearest_ids = torch.matmul(pseudo_norm, token_bank_norm.t()).topk(
            k=num_pic2word_tokens, dim=-1
        ).indices
        nearest_token_embeddings = token_bank[nearest_ids]
        query_image_tokens = tuple(
            nearest_token_embeddings[:, i, :] for i in range(num_pic2word_tokens)
        )
    composed_feature = model.encode_text_img_retrieval(
        text_tokens,
        query_image_tokens,
        split_ind=split_token_id,
        repeat=False,
    )
    return composed_feature

def _maybe_log_embedding_topk_tokens(nearest_ids, args):
    if not getattr(args, "embedding_feature_log_words", False):
        return

    logged_batches = int(getattr(args, "_embedding_feature_logged_batches", 0))
    max_batches = max(1, int(getattr(args, "embedding_feature_log_max_batches", 1)))
    if logged_batches >= max_batches:
        return

    tokenizer = getattr(open_clip, "_tokenizer", None)
    decoder = getattr(tokenizer, "decoder", None) if tokenizer is not None else None
    if decoder is None:
        logging.info("embedding_feature_log_words is enabled, but tokenizer decoder is unavailable.")
        setattr(args, "_embedding_feature_logged_batches", logged_batches + 1)
        return

    raw_log_topk = getattr(args, "embedding_feature_log_topk", None)
    if raw_log_topk is None:
        raw_log_topk = nearest_ids.size(1)
    topk_to_show = min(nearest_ids.size(1), max(1, int(raw_log_topk)))
    samples_to_show = min(
        nearest_ids.size(0),
        max(1, int(getattr(args, "embedding_feature_log_samples", 2))),
    )

    for sample_idx in range(samples_to_show):
        token_ids = nearest_ids[sample_idx, :topk_to_show].detach().cpu().tolist()
        token_strs = [
            decoder.get(int(token_id), f"<id:{int(token_id)}>").replace("</w>", "").strip()
            for token_id in token_ids
        ]
        logging.info(
            "[EmbeddingTopK] sample=%d token_ids=%s tokens=%s",
            sample_idx,
            token_ids,
            token_strs,
        )

    setattr(args, "_embedding_feature_logged_batches", logged_batches + 1)


def encode_embedding_topk_feature(model, img2text, images, texts, args):
    """
    Build embedding feature by:
      1) predicting pseudo word embedding from image,
      2) selecting top-k nearest token embeddings from CLIP vocab,
      3) replacing the marker token with these k embeddings,
      4) encoding the composed text feature for retrieval.
    """
    text_tokens = tokenize_to_device(texts, args)
    split_text = getattr(args, "global_flow_pic2word_marker", "*")
    split_token_id = tokenize([split_text])[0][1].item()
    embedding_topk_text = getattr(args, "embedding_feature_topk_text", None)
    if embedding_topk_text is None:
        embedding_topk_text = getattr(args, "global_flow_pic2word_topk_text", 1)
    topk_text = max(1, int(embedding_topk_text))
    text_tokens = _expand_pic2word_marker_slots(text_tokens, split_token_id, topk_text)

    if not torch.all((text_tokens == split_token_id).any(dim=1)):
        raise ValueError(
            "embedding feature requires every text prompt to "
            f"contain the marker token {split_text!r}."
        )

    image_features = model.encode_image(images)
    pseudo_word_embedding = img2text(image_features)
    token_bank = model.token_embedding.weight.type(model.dtype)
    pseudo_norm = F.normalize(pseudo_word_embedding.type(model.dtype), dim=-1)
    token_bank_norm = F.normalize(token_bank, dim=-1)
    nearest_ids = torch.matmul(pseudo_norm, token_bank_norm.t()).topk(
        k=topk_text, dim=-1
    ).indices
    _maybe_log_embedding_topk_tokens(nearest_ids, args)
    nearest_token_embeddings = token_bank[nearest_ids]
    query_image_tokens = tuple(
        nearest_token_embeddings[:, i, :] for i in range(topk_text)
    )
    embedding_feature = model.encode_text_img_retrieval(
        text_tokens,
        query_image_tokens,
        split_ind=split_token_id,
        repeat=False,
    )
    return embedding_feature

def encode_image_batch(model, images, args):
    """
    Standard text encoding with CLIP text encoder.
    texts -> token ids -> text features
    """
    image_features = model.encode_image(images)
    return image_features
    
def build_global_flow_feature(model, img2text, ref_images, texts, args, source, text_weight, image_weight):
    source = source.lower()
    if source == "text":
        feature = encode_text_batch(model, texts, args)
    elif source == "image":
        feature = encode_image_batch(model, ref_images, args)
    elif source == "inversion":
        feature = encode_image_via_img2text(model, img2text, ref_images, args)
    elif source == "composed":
        compose_method = getattr(args, "global_flow_compose_method", "add").lower()
        if compose_method == "pic2word":
            feature = encode_pic2word_composed_feature(model, img2text, ref_images, texts, args)
        else:
            text_feature = encode_text_batch(model, texts, args)
            image_feature = encode_image_via_img2text(model, img2text, ref_images, args)
            feature = text_weight * text_feature + image_weight * image_feature
            if compose_method == "mean":
                denom = max(text_weight + image_weight, 1e-6)
                feature = feature / denom
            elif compose_method != "add":
                raise ValueError(f"Unsupported global_flow_compose_method: {compose_method}")
    else:
        raise ValueError(f"Unsupported global flow feature source: {source}")

    return _normalize_feature(feature)


def build_text_mask(tokens):
    return tokens.ne(0)


def _transformer_forward_until(transformer, x, end_layer):
    if hasattr(transformer, "forward_until"):
        return transformer.forward_until(x, end_layer=end_layer)

    blocks = transformer.resblocks
    if end_layer is None:
        end_layer = len(blocks)
    if end_layer < 0:
        end_layer = len(blocks) + end_layer
    end_layer = max(0, min(end_layer, len(blocks)))
    for block in blocks[:end_layer]:
        x = block(x)
    return x


def _transformer_forward_from(transformer, x, start_layer):
    if hasattr(transformer, "forward_from"):
        return transformer.forward_from(x, start_layer=start_layer)

    blocks = transformer.resblocks
    if start_layer < 0:
        start_layer = len(blocks) + start_layer
    start_layer = max(0, min(start_layer, len(blocks)))
    for block in blocks[start_layer:]:
        x = block(x)
    return x


def extract_text_token_features(model, text, end_layer=-1):
    if hasattr(model, "encode_text_tokens"):
        return model.encode_text_tokens(text, end_layer=end_layer)

    x = model.token_embedding(text).type(model.dtype)
    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)
    x = _transformer_forward_until(model.transformer, x, end_layer=end_layer)
    x = x.permute(1, 0, 2)
    return x


def extract_image_token_features(model, images, end_layer=-1):
    if hasattr(model, "encode_image_tokens"):
        return model.encode_image_tokens(images, end_layer=end_layer)

    visual = model.visual
    if hasattr(visual, "get_intermediate_tokens"):
        return visual.get_intermediate_tokens(images.type(model.dtype), end_layer=end_layer)

    raise AttributeError("Current CLIP visual encoder does not expose intermediate visual tokens.")


def encode_text_from_token_features(
    model,
    text,
    token_features,
    start_layer=-1,
    pooling="eot",
    ref_token_features=None,
    pooling_k=3,
):
    x = token_features.type(model.dtype)
    x = x.permute(1, 0, 2)
    x = _transformer_forward_from(model.transformer, x, start_layer=start_layer)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)

    valid_mask = text.ne(0)

    if pooling == "eot":
        end_id = getattr(model, "end_id", model.vocab_size - 1)
        collect_ind = (text == end_id).nonzero()[:, 1]
        pooled = x[torch.arange(x.size(0), device=x.device), collect_ind]
    elif pooling == "last_valid":
        collect_ind = valid_mask.long().sum(dim=1).sub(1).clamp_min(0)
        pooled = x[torch.arange(x.size(0), device=x.device), collect_ind]
    elif pooling == "mean":
        mean_mask = valid_mask.unsqueeze(-1)
        denom = mean_mask.sum(dim=1).clamp_min(1)
        pooled = (x * mean_mask).sum(dim=1) / denom
    elif pooling == "tailk_mean":
        tail_scores = valid_mask.float()
        tail_scores = tail_scores + torch.arange(
            text.size(1),
            device=text.device,
            dtype=x.dtype,
        ).unsqueeze(0) / max(text.size(1), 1)
        topk = min(max(int(pooling_k), 1), text.size(1))
        tail_idx = tail_scores.topk(topk, dim=1).indices
        tail_hidden = x.gather(1, tail_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        tail_mask = valid_mask.gather(1, tail_idx).unsqueeze(-1).type_as(tail_hidden)
        pooled = (tail_hidden * tail_mask).sum(dim=1) / tail_mask.sum(dim=1).clamp_min(1.0)
    elif pooling == "topk_changed":
        if ref_token_features is None:
            raise ValueError("topk_changed pooling requires ref_token_features.")
        ref_x = ref_token_features.type(model.dtype)
        ref_x = ref_x.permute(1, 0, 2)
        ref_x = _transformer_forward_from(model.transformer, ref_x, start_layer=start_layer)
        ref_x = ref_x.permute(1, 0, 2)
        ref_x = model.ln_final(ref_x).type(model.dtype)

        change = 1.0 - F.cosine_similarity(x, ref_x, dim=-1)
        change = change.masked_fill(~valid_mask, float("-inf"))
        topk = min(max(int(pooling_k), 1), text.size(1))
        changed_idx = change.topk(topk, dim=1).indices
        changed_hidden = x.gather(1, changed_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        changed_mask = valid_mask.gather(1, changed_idx).unsqueeze(-1).type_as(changed_hidden)
        pooled = (changed_hidden * changed_mask).sum(dim=1) / changed_mask.sum(dim=1).clamp_min(1.0)
    else:
        raise ValueError(f"Unsupported text pooling mode: {pooling}")

    x = pooled @ model.text_projection
    return x

def flow_matching_inference(
    flow_net,
    q,
    e_m=None,
    num_steps=4,
    training_objective="flow_matching",
    eps=1e-6,
    step_normalize=True,
    step_norm_type="l2",
    hybrid_geodesic_steps=0,
):
    def l2norm(x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)

    def project_to_tangent(x, v):
        x_unit = l2norm(x)
        return v - (v * x_unit).sum(dim=-1, keepdim=True) * x_unit

    def sphere_expmap_step(x, v, dt):
        x_unit = l2norm(x)
        v_tan = project_to_tangent(x_unit, v)
        speed = v_tan.norm(dim=-1, keepdim=True).clamp(min=eps)
        theta = dt * speed
        direction = v_tan / speed
        x_next = torch.cos(theta) * x_unit + torch.sin(theta) * direction
        return l2norm(x_next)

    q = l2norm(q)
    if e_m is not None:
        e_m = l2norm(e_m)

    x_t = q.clone()
    x0 = q.clone()
    B = q.size(0)
    if training_objective == "start_end_mse":
        t0 = torch.zeros((B, 1), device=q.device, dtype=q.dtype)
        return torch.tanh(flow_net(q, delta=torch.zeros_like(q), e_m=e_m, t=t0))

    dt = 1.0 / num_steps
    hybrid_geodesic_steps = max(0, min(int(hybrid_geodesic_steps), int(num_steps)))

    for k in range(num_steps):
        t = torch.full(
            (B, 1),
            k / num_steps,
            device=q.device,
            dtype=q.dtype,
        )
        delta = x_t - x0
        v = flow_net(x_t, delta=delta, e_m=e_m, t=t)
        v = torch.tanh(v)
        if hybrid_geodesic_steps > 0:
            if k < hybrid_geodesic_steps:
                v_tan = project_to_tangent(x_t, v)
                if step_norm_type == "expmap":
                    x_t = sphere_expmap_step(x_t, v_tan, dt=dt)
                else:
                    x_t = l2norm(x_t + dt * v_tan)
            else:
                x_t = x_t + dt * v
        else:
            if step_normalize:
                v_tan = project_to_tangent(x_t, v)
                if step_norm_type == "expmap":
                    x_t = sphere_expmap_step(x_t, v_tan, dt=dt)
                else:
                    x_t = l2norm(x_t + dt * v_tan)
            else:
                x_t = x_t + dt * v

    return x_t


def sequence_flow_matching_inference(
    flow_net,
    src_tokens,
    vis_tokens,
    src_mask=None,
    vis_mask=None,
    num_steps=4,
):
    x_t = src_tokens.clone()
    batch_size = src_tokens.size(0)
    dt = 1.0 / num_steps

    if src_mask is None:
        src_mask = torch.ones(
            src_tokens.shape[:2],
            device=src_tokens.device,
            dtype=torch.bool,
        )
    if vis_mask is None:
        vis_mask = torch.ones(
            vis_tokens.shape[:2],
            device=vis_tokens.device,
            dtype=torch.bool,
        )

    x_mask = torch.ones_like(src_mask, dtype=torch.bool)

    for k in range(num_steps):
        t = torch.full(
            (batch_size, 1),
            k / num_steps,
            device=src_tokens.device,
            dtype=src_tokens.dtype,
        )
        flow_output = flow_net(
            x_t=x_t,
            src_tokens=src_tokens,
            vis_tokens=vis_tokens,
            t=t,
            src_mask=src_mask,
            vis_mask=vis_mask,
            x_mask=x_mask,
        )
        x_t = x_t + dt * flow_output.velocity

    return x_t
    


def prepare_img(img_file, transform):
    return transform(Image.open(img_file))

def visualize_results(model, img2text, args, prompt, dataloader):        
    model.eval()
    img2text.eval()   
    if not os.path.exists(args.demo_out):
        os.makedirs(args.demo_out)        
    if not os.path.exists(os.path.join(args.demo_out, "images")):
        os.makedirs(os.path.join(args.demo_out, "images"))
    text = []
    id_split = tokenize(["*"])[0][1]
    for p in prompt:
        text_tokens = tokenize(p)
        text.append(text_tokens)
        assert id_split in text_tokens
    text = torch.cat(text, dim=0)    
    text = text.cuda(args.gpu, non_blocking=True)    
    all_image_features, all_image_filenames = [], []
    m = model.module if args.distributed or args.dp else model
    query_file = args.query_file
    path_save = os.path.join("./data", args.retrieval_data.split('/')[-1].split('.')[0]+".pkl")
    if os.path.exists(path_save):
        with open(path_save, 'rb') as f:
            data = pickle.load(f)
        all_image_features = data['feats']
        all_image_filenames = data['path']
        all_image_features = torch.from_numpy(all_image_features).cuda(args.gpu, non_blocking=True)
    else:
        ## Extract features of target images. 
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, filenames = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                image_features = m.encode_image(images)           
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
                all_image_features.append(image_features)
                for name in filenames:
                    all_image_filenames.append(name)
            all_image_features = torch.cat(all_image_features, dim=0)
            dict_save = {}
            dict_save['feats'] = all_image_features.data.cpu().numpy()
            dict_save['path'] = all_image_filenames
            with open(path_save,"wb") as f:
                pickle.dump(dict_save,f)
    f = open(os.path.join(args.demo_out, "index.html"), 'w')
    html_txt = """"""
    ## For each domain, compute composed features and evaluate.
    for query in query_file.split(","):        
        logging.info("retrieve image of {}".format(query))
        transform = _transform(model.visual.input_resolution)
        query_img = prepare_img(query, transform)
        query_img = torch.unsqueeze(query_img, 0)    
        query_img = query_img.cuda(args.gpu, non_blocking=True)
        img_feature = m.encode_image(query_img) 
        query_img_feature = img2text(img_feature)
        composed_feature = m.encode_text_img_vis(text, query_img_feature, split_ind=id_split)
        composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
        img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)
        text_feature = m.encode_text(text)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        similarity = composed_feature @ all_image_features.T
        _, indices = torch.sort(similarity, descending=True)        
        logging.info("Composed feature result")
        for i, caption in enumerate(prompt):
            logging.info("for prompt {}".format(caption))
            for j, ind in enumerate(indices[i][:8]):
                logging.info("top {} filename {}".format(j, all_image_filenames[ind]))
        image_paths = [[all_image_filenames[ind] for j, ind in enumerate(indices[i][:8])] 
                        for i, caption in enumerate(prompt)]
        html_txt += make_html(prompt, query, image_paths, args.demo_out)
    f.write(html_txt)

def make_html(prompts, query_image, images, path_html):
    import shutil
    html_all = """"""        
    for i in range(len(prompts)):
        prompt = prompts[i]            
        query_image_local = os.path.join(path_html, "images", query_image.split("/")[-1])
        query_image_local_path = os.path.join("images", query_image.split("/")[-1])
        shutil.copy(query_image, query_image_local)
        image_list = images[i]        
        html = """<table><tr>"""    
        html += """<td><p style="display:inline-block;vertical-align;font-size:20px">%s</p></td>"""%(prompt)
        html += """<td><p style="margin-right: 50px;"><img src="%s" height="100"></p></td>"""%(query_image_local_path)
        for image in image_list:
            image_local = os.path.join(path_html, "images", image.split("/")[-1])
            image_path = os.path.join("images", image.split("/")[-1])
            shutil.copy(image, image_local)
            html += """<td><img src="%s" height=%s></td>"""%(image_path, 200)
        html += """</tr></table>"""
        html_all += html
    return html_all
    #f.write(html_all)


def evaluate_imgnet_retrieval(model, img2text, args, prompt, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    all_image_features = []  
    all_target_labels = []      
    m = model.module if args.distributed or args.dp else model
    n_class = 1000
   
    with torch.no_grad():
        ## Extract target image features. 
        for batch in tqdm(target_loader):
            images, labels = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            all_image_features.append(image_features)
            all_target_labels.append(labels)
            logit_scale = m.logit_scale.exp()
            logit_scale = logit_scale.mean()   

        ## Extract query features 
        for p_ind, p in enumerate(prompt):            
            ## which token has to be replaced with image features
            id_split = tokenize(["*"])[0][1]
            text = tokenize(p).view(1, -1)
            text = text.cuda(args.gpu, non_blocking=True)
            ## text only features (domain name only)
            text_only = p.replace("*", "")
            text_only = tokenize(text_only).view(1, -1)            
            text_only = text_only.cuda(args.gpu, non_blocking=True)                        
            text_only_features = m.encode_text(text_only)
            text_only_features_normed = text_only_features / text_only_features.norm(dim=-1, keepdim=True)

            all_query_features = []
            all_query_image_features = []
            all_query_mixture_features = []
            all_query_labels = []
            all_text_features = []
            for batch in tqdm(query_loader):
                images, labels = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                    labels = labels.cuda(args.gpu, non_blocking=True)
                ## Label is decided by class label and images' domain
                labels += n_class * p_ind
                image_features = m.encode_image(images)
                 ## Composed feature extraction
                image_features_query = img2text(image_features)                      
                composed_feature = m.encode_text_img_retrieval(text, image_features_query, split_ind=id_split)                            
                composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
                ## Image feature only
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
                ## average of image and text features
                mixture_features = image_features + text_only_features_normed
                mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)       

                all_text_features.append(text_only_features_normed.repeat((image_features.shape[0], 1)))
                all_query_features.append(composed_feature)
                all_query_image_features.append(image_features)
                all_query_mixture_features.append(mixture_features)
                all_query_labels.append(labels)

            metric_func = partial(get_metrics_imgnet, 
                image_features=torch.cat(all_image_features), 
                query_labels=torch.cat(all_query_labels),
                target_labels=torch.cat(all_target_labels),
                )

            feats = {'composed': torch.cat(all_query_features), 
                    'image': torch.cat(all_query_image_features),
                    'text': torch.cat(all_text_features),
                    'mixture': torch.cat(all_query_mixture_features)}        

            for key, value in feats.items():
                metrics = metric_func(query_features=value)
                logging.info(
                f"Eval {key} Feature"
                + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    return metrics


def evaluate_coco(model, img2text, args, loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_mixture_features = []  
    all_composed_features_with_class = []  
    all_text_full_features = [] 

    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()
    with torch.no_grad():
        for batch in tqdm(loader):
            images, region_images, text_full, text_with_blank, text_with_blank_query, filename, raw_text = batch            
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                region_images = region_images.cuda(args.gpu, non_blocking=True)
                text_full = text_full.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                text_with_blank_query = text_with_blank_query.cuda(args.gpu, non_blocking=True)

            ## Target image features 
            image_features = m.encode_image(images)             
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
            id_split = tokenize(["*"])[0][1]
            ## Composed image features
            query_image_features = m.encode_image(region_images)
            query_image_tokens = img2text(query_image_features)          
            composed_feature_with_class = m.encode_text_img_retrieval(text_with_blank_query, query_image_tokens, split_ind=id_split, repeat=False)                        
            composed_feature_with_class = composed_feature_with_class / composed_feature_with_class.norm(dim=-1, keepdim=True)        
            ## Text only features
            text_full_features = m.encode_text(text_full)
            text_full_features = text_full_features / text_full_features.norm(dim=-1, keepdim=True)            
            ## Query only features
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)                               
            ## Mixed featurs
            mixture_features = query_image_features + text_full_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)            

            all_image_features.append(image_features.cpu())
            all_text_full_features.append(text_full_features.cpu())       
            all_query_image_features.append(query_image_features.cpu())
            all_mixture_features.append(mixture_features.cpu())                        
            all_composed_features_with_class.append(composed_feature_with_class.cpu())            

        metric_func = partial(get_metrics_coco, 
                image_features=torch.cat(all_image_features), 
                logit_scale=logit_scale
                )
        feats = {'composed': torch.cat(all_composed_features_with_class), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_text_full_features),
                 'mixture': torch.cat(all_mixture_features)}        

        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    return metrics


def evaluate_cirr(model, img2text, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_mixture_features = []  
    all_caption_features = []  
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []
    all_raw_captions = []
    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()   

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

        for batch in tqdm(query_loader):
            ref_images, text_with_blank, caption_only, ref_paths, answer_paths, raw_captions = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            id_split = tokenize(["*"])[0][1]                        
            for path in ref_paths:
                all_ref_paths.append(path)
            for path in answer_paths:
                all_answer_paths.append(path)
            for cap in raw_captions:
                all_raw_captions.append(cap)

            caption_features = m.encode_text(caption_only)
            ## Composed features
            query_image_features = m.encode_image(ref_images)
            query_image_tokens = img2text(query_image_features)
            composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)                

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
            mixture_features = query_image_features + caption_features            
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)                        

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)
        
        metric_func = partial(get_metrics_cirr, 
                image_features=torch.cat(all_image_features), 
                reference_names=all_ref_paths, 
                index_names=all_target_paths, 
                target_names=all_answer_paths)

        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}
        
        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return metrics


def evaluate_cirr_fm(model, img2text, args, query_loader, target_loader, flow_net=None):
    """
    CIRR evaluation with optional flow-matching feature branch.
    Keeps legacy baselines (composed/image/text/mixture) and adds `flow` when flow_net is provided.
    """
    if not is_master(args):
        return

    model.eval()
    img2text.eval()
    if flow_net is not None:
        flow_net.eval()

    all_image_features = []
    all_query_image_features = []
    all_composed_features = []
    all_embedding_features = []
    all_flow_features = []
    all_mixture_features = []
    all_caption_features = []
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []

    m = unwrap_model(model)
    it = unwrap_model(img2text)
    fm = unwrap_model(flow_net) if flow_net is not None else None

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

        for batch in tqdm(query_loader):
            ref_images, text_with_blank, caption_only, ref_paths, answer_paths, _ = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)

            id_split = tokenize(["*"])[0][1]
            for path in ref_paths:
                all_ref_paths.append(path)
            for path in answer_paths:
                all_answer_paths.append(path)

            caption_features = m.encode_text(caption_only)
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)

            query_image_features = m.encode_image(ref_images)
            

            query_image_tokens = it(query_image_features)
            composed_feature = m.encode_text_img_retrieval(
                text_with_blank, query_image_tokens, split_ind=id_split, repeat=False
            )
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)

            embedding_feature = encode_embedding_topk_feature(
                m, it, ref_images, text_with_blank, args
            )
            embedding_feature = embedding_feature / embedding_feature.norm(dim=-1, keepdim=True)
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)

            if fm is not None:
                if getattr(args, "loss_type", "global") == "sequence":
                    src_tokens = extract_text_token_features(m, text_with_blank, end_layer=-1)
                    vis_tokens = extract_image_token_features(m, ref_images, end_layer=-1)

                    if getattr(args, "seq_flow_drop_visual_cls", False):
                        vis_tokens = vis_tokens[:, 1:, :]

                    if getattr(args, "seq_flow_token_norm", False):
                        src_tokens = F.normalize(src_tokens, dim=-1)
                        vis_tokens = F.normalize(vis_tokens, dim=-1)

                    flow_tokens = sequence_flow_matching_inference(
                        fm,
                        src_tokens=src_tokens,
                        vis_tokens=vis_tokens,
                        src_mask=build_text_mask(text_with_blank),
                        num_steps=getattr(args, "flow_num_steps", 16),
                    )
                    flow_feature = encode_text_from_token_features(
                        m,
                        text_with_blank,
                        flow_tokens,
                        start_layer=-1,
                        pooling=getattr(args, "seq_flow_pooling", "eot"),
                        ref_token_features=src_tokens,
                        pooling_k=getattr(args, "seq_flow_pooling_k", 3),
                    )
                else:
                    q = build_global_flow_feature(
                        model=m,
                        img2text=it,
                        ref_images=ref_images,
                        texts=text_with_blank,
                        args=args,
                        source=getattr(args, "global_flow_start_source", "text"),
                        text_weight=getattr(args, "global_flow_start_text_weight", 1.0),
                        image_weight=getattr(args, "global_flow_start_image_weight", 1.0),
                    )
                    use_condition = getattr(args, "global_flow_conditioning", "enabled") == "enabled"
                    e_m = None
                    if use_condition:
                        e_m = build_global_flow_feature(
                            model=m,
                            img2text=it,
                            ref_images=ref_images,
                            texts=caption_only,
                            args=args,
                            source=getattr(args, "global_flow_condition_source", "image"),
                            text_weight=getattr(args, "global_flow_condition_text_weight", 1.0),
                            image_weight=getattr(args, "global_flow_condition_image_weight", 1.0),
                        )
                    q = apply_global_start_noise(q, args)
                    q = _normalize_feature(q)
                    if e_m is not None:
                        e_m = _normalize_feature(e_m)

                    flow_feature = flow_matching_inference(
                        fm,
                        q,
                        e_m,
                        num_steps=getattr(args, "flow_num_steps", 16),
                        training_objective=getattr(args, "flow_training_objective", "flow_matching"),
                        step_normalize=getattr(args, "flow_step_normalize", True),
                        step_norm_type=getattr(args, "flow_step_norm_type", "l2"),
                        hybrid_geodesic_steps=getattr(args, "flow_hybrid_geodesic_steps", 0),
                    )

                flow_feature = flow_feature / flow_feature.norm(dim=-1, keepdim=True)
                all_flow_features.append(flow_feature)

            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)
            all_embedding_features.append(embedding_feature)
            all_mixture_features.append(mixture_features)

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)

        metric_func = partial(
            get_metrics_cirr,
            image_features=torch.cat(all_image_features),
            reference_names=all_ref_paths,
            index_names=all_target_paths,
            target_names=all_answer_paths,
        )

        feats = {
            "composed": torch.cat(all_composed_features),
            "embedding": torch.cat(all_embedding_features),
            "image": torch.cat(all_query_image_features),
            "text": torch.cat(all_caption_features),
            "mixture": torch.cat(all_mixture_features),
        }
        if len(all_flow_features) > 0:
            feats["flow"] = torch.cat(all_flow_features)

        metrics_by_feature = {}
        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            metrics_by_feature[key] = metrics
            logging.info(
                f"Eval {key} Feature" + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            )

    return metrics_by_feature


def evaluate_cirr_test(model, img2text, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_composed_plus_image_features = []  
    all_mixture_features = []  
    all_caption_features = []  
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []
    all_ids = []

    m = model.module if args.distributed or args.dp else model   
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()   

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

        for batch in tqdm(query_loader):
            ref_images, text_with_blank, caption_only, ref_paths, pairids = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            id_split = tokenize(["*"])[0][1]                        
            for ids in pairids:
                all_ids.append(ids)
            for path in ref_paths:
                all_ref_paths.append(path)

            caption_features = m.encode_text(caption_only)
            query_image_features = m.encode_image(ref_images)

            if args.eval_combiner:
                composed_feature = img2text(query_image_features, caption_features)
            else:
                query_image_tokens = img2text(query_image_features)
                composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)            

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)
        res_all = {}
        metrics_func = partial(get_cirr_testoutput, 
                               image_features=torch.cat(all_image_features),
                               reference_names=all_ref_paths,
                               index_names=all_target_paths,
                               id_names=all_ids)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}        
        for key, value in feats:
            res_all[key] = metrics_func(ref_features=value)
    return res_all


def evaluate_fashion(model, img2text, args, source_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    all_target_paths = []
    all_answer_paths = []
    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_caption_features = []  
    all_mixture_features = []  
    all_reference_names = []
    all_captions = []     
    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean() 

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

    with torch.no_grad():
        for batch in tqdm(source_loader):
            ref_images, target_images, target_caption, caption_only, answer_paths, ref_names, captions = batch
            for path in answer_paths:
                all_answer_paths.append(path)
            all_reference_names.extend(ref_names)
            all_captions.extend(captions)
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                target_images = target_images.cuda(args.gpu, non_blocking=True)
                target_caption = target_caption.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            query_image_features = m.encode_image(ref_images)
            id_split = tokenize(["*"])[0][1]            
            caption_features = m.encode_text(target_caption)                            
            query_image_tokens = img2text(query_image_features)          
            composed_feature = m.encode_text_img_retrieval(target_caption, query_image_tokens, split_ind=id_split, repeat=False)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)

            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)                         

        metric_func = partial(get_metrics_fashion, 
                              image_features=torch.cat(all_image_features),
                              target_names=all_target_paths, answer_names=all_answer_paths)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}
        
        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return metrics


def get_metrics_coco(image_features, ref_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale.cpu() * image_features @ ref_features.t()).detach().cpu()
    logits_per_ref = logits_per_image.t().detach().cpu()
    logits = {"image_to_ref": logits_per_image, "ref_to_image": logits_per_ref}
    ground_truth = torch.arange(len(ref_features)).view(-1, 1)
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10, 50, 100]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    return metrics


def get_metrics_fashion(image_features, ref_features, target_names, answer_names):
    metrics = {}
    # distances = 1 - ref_features @ image_features.T    
    # sorted_indices = torch.argsort(distances, dim=-1).cpu()
    # sorted_index_names = np.array(target_names)[sorted_indices]
    # labels = torch.tensor(
    #     sorted_index_names == np.repeat(np.array(answer_names), len(target_names)).reshape(len(answer_names), -1))
    # assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(answer_names)).int())
    # # Compute the metrics
    # for k in [1, 5, 10, 50, 100]:
    #     metrics[f"R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
    query_features = F.normalize(ref_features.detach().cpu(), dim=-1)
    gallery_features = F.normalize(image_features.detach().cpu(), dim=-1)

    logits = query_features @ gallery_features.t()
    ranking = torch.argsort(logits, dim=-1, descending=True)

    name_to_index = {name: idx for idx, name in enumerate(target_names)}
    gt_index = torch.tensor([name_to_index[name] for name in answer_names], dtype=torch.long)
    gt_pos = torch.where(ranking == gt_index.view(-1, 1))[1]
    for k in [1, 5, 10, 50, 100]:
        k_eff = min(k, ranking.size(1))
        metrics[f"R@{k}"] = (gt_pos < k_eff).float().mean().item() * 100.0
    return metrics


def get_metrics_cirr(image_features, ref_features, reference_names, index_names, target_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), 
        len(index_names)).reshape(len(target_names), -1))        
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), 
        len(index_names) - 1).reshape(len(target_names), -1))

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    for k in [1, 5, 10, 50, 100]:
        metrics[f"recall_R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100

    return metrics


def get_cirr_testoutput(image_features, ref_features, reference_names, index_names, id_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    result_dict = {"version": "rc2", "metric": "recall"}
    for ind in range(len(id_names)):
        pairid = str(id_names[ind].item())
        result_dict[pairid] = []
        for t in range(50):
            result_dict[pairid].append(sorted_index_names[ind][t].replace(".png", ""))
    return result_dict


def get_metrics_imgnet(query_features, image_features, query_labels, target_labels):
    metrics = {}
    num_classes = 7000
    query_onehot = F.one_hot(query_labels, num_classes=num_classes).float()
    target_onehot = F.one_hot(target_labels, num_classes=num_classes).float()
    batches = [(query_features[x:x+100], query_onehot[x:x+100]) for x in range(0, len(query_features), 100)]
    for k in [1, 5, 10, 50, 100, 200]:
        metrics[f"Real2Sketch_R@{k}"] = 0
        metrics[f"Real2Sketch_P@{k}"] = 0
    for batch in batches:
        feats, labels = batch[0], batch[1]
        logits_per_query = (feats @ image_features.t()).detach().cpu()
        label_matrix = (labels @ target_onehot.t()).detach().cpu()                
        ranking = torch.argsort(logits_per_query, descending=True)
        for k in [1, 5, 10, 50, 100, 200]:
            matrix_k = torch.zeros_like(label_matrix)
            rank_k = ranking[:, :k]
            matrix_k[torch.arange(matrix_k.size(0)).unsqueeze(1), rank_k] = 1
            consistency = matrix_k * label_matrix
            num_correct = torch.sum(consistency, dim=1)
            num_predicted = torch.sum(matrix_k, dim=1)            
            num_total = torch.sum(label_matrix, dim=1)
            recall = torch.mean(num_correct / (num_total+1e-5))
            precision = torch.mean(num_correct / num_predicted)
            metrics[f"Real2Sketch_R@{k}"] += recall * len(feats)
            metrics[f"Real2Sketch_P@{k}"] += precision * len(feats)
    for k in [1, 5, 10, 50, 100, 200]:
        metrics[f"Real2Sketch_R@{k}"] /= len(query_features)
        metrics[f"Real2Sketch_P@{k}"] /= len(query_features)
    return metrics

def evaluate_fashion_fm(model, img2text, args, source_loader, target_loader, flow_net=None):
    if not is_master(args):
        return

    model.eval()
    img2text.eval()
    if flow_net is not None:
        flow_net.eval()

    all_target_paths = []
    all_answer_paths = []
    all_image_features = []
    all_query_image_features = []
    all_composed_features = []
    all_embedding_features = []
    all_flow_features = []
    all_caption_features = []
    all_mixture_features = []
    all_reference_names = []
    all_captions = []

    m = unwrap_model(model)
    it = unwrap_model(img2text)
    fm = unwrap_model(flow_net) if flow_net is not None else None

    with torch.no_grad():
        # target gallery features
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)

            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)

            for path in target_paths:
                all_target_paths.append(path)

        # query features
        for batch in tqdm(source_loader):
            ref_images, target_images, target_caption, caption_only, answer_paths, ref_names, captions = batch

            for path in answer_paths:
                all_answer_paths.append(path)
            all_reference_names.extend(ref_names)
            all_captions.extend(captions)

            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                target_images = target_images.cuda(args.gpu, non_blocking=True)
                target_caption = target_caption.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)

            # target image feature only used for analysis/debug
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # ref image feature
            query_image_features = m.encode_image(ref_images)
            

            # old composed baseline
            id_split = tokenize(["*"])[0][1]
            caption_features = m.encode_text(target_caption)
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)

            query_image_tokens = it(query_image_features)
            composed_feature = m.encode_text_img_retrieval(
                target_caption, query_image_tokens, split_ind=id_split, repeat=False
            )
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)

            embedding_feature = encode_embedding_topk_feature(
                m, it, ref_images, target_caption, args
            )
            embedding_feature = embedding_feature / embedding_feature.norm(dim=-1, keepdim=True)

            # mixture baseline
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)

            # -----------------------------
            # Flow Matching feature
            # q   = ref image -> img2text -> text encoder
            # e_m = modification text -> text encoder
            # y^  = flow(q, e_m)
            # -----------------------------
            if fm is not None:
                if getattr(args, "loss_type", "global") == "sequence":
                    src_tokens = extract_text_token_features(m, caption_only, end_layer=-1)
                    vis_tokens = extract_image_token_features(m, ref_images, end_layer=-1)

                    if getattr(args, "seq_flow_drop_visual_cls", False):
                        vis_tokens = vis_tokens[:, 1:, :]

                    if getattr(args, "seq_flow_token_norm", False):
                        src_tokens = F.normalize(src_tokens, dim=-1)
                        vis_tokens = F.normalize(vis_tokens, dim=-1)

                    flow_tokens = sequence_flow_matching_inference(
                        fm,
                        src_tokens=src_tokens,
                        vis_tokens=vis_tokens,
                        src_mask=build_text_mask(caption_only),
                        num_steps=getattr(args, "flow_num_steps", 16),
                    )
                    flow_feature = encode_text_from_token_features(
                        m,
                        caption_only,
                        flow_tokens,
                        start_layer=-1,
                        pooling=getattr(args, "seq_flow_pooling", "eot"),
                        ref_token_features=src_tokens,
                        pooling_k=getattr(args, "seq_flow_pooling_k", 3),
                    )
                else:
                    q = build_global_flow_feature(
                        model=m,
                        img2text=it,
                        ref_images=ref_images,
                        texts=caption_only,
                        args=args,
                        source=getattr(args, "global_flow_start_source", "text"),
                        text_weight=getattr(args, "global_flow_start_text_weight", 1.0),
                        image_weight=getattr(args, "global_flow_start_image_weight", 1.0),
                    )
                    use_condition = getattr(args, "global_flow_conditioning", "enabled") == "enabled"
                    e_m = None
                    if use_condition:
                        e_m = build_global_flow_feature(
                            model=m,
                            img2text=it,
                            ref_images=ref_images,
                            texts=caption_only,
                            args=args,
                            source=getattr(args, "global_flow_condition_source", "image"),
                            text_weight=getattr(args, "global_flow_condition_text_weight", 1.0),
                            image_weight=getattr(args, "global_flow_condition_image_weight", 1.0),
                        )
                    q = apply_global_start_noise(q, args)
                    q = _normalize_feature(q)
                    if e_m is not None:
                        e_m = _normalize_feature(e_m)

                    flow_feature = flow_matching_inference(
                        fm,
                        q,
                        e_m,
                        num_steps=getattr(args, "flow_num_steps", 16),
                        training_objective=getattr(args, "flow_training_objective", "flow_matching"),
                        step_normalize=getattr(args, "flow_step_normalize", True),
                        step_norm_type=getattr(args, "flow_step_norm_type", "l2"),
                        hybrid_geodesic_steps=getattr(args, "flow_hybrid_geodesic_steps", 0),
                    )
                flow_feature = flow_feature / flow_feature.norm(dim=-1, keepdim=True)
                all_flow_features.append(flow_feature)

            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)
            all_embedding_features.append(embedding_feature)
            all_mixture_features.append(mixture_features)

        metric_func = partial(
            get_metrics_fashion,
            image_features=torch.cat(all_image_features),
            target_names=all_target_paths,
            answer_names=all_answer_paths
        )

        feats = {
            'composed': torch.cat(all_composed_features),
            'embedding': torch.cat(all_embedding_features),
            'image': torch.cat(all_query_image_features),
            'text': torch.cat(all_caption_features),
            'mixture': torch.cat(all_mixture_features),
        }

        if len(all_flow_features) > 0:
            feats['flow'] = torch.cat(all_flow_features)

        metrics_by_feature = {}
        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            metrics_by_feature[key] = metrics
            logging.info(
                f"Eval {key} Feature\t" +
                "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            )

    return metrics_by_feature
