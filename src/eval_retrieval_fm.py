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

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*"
)

import os
import time
import logging
from time import gmtime, strftime
from pathlib import Path
import json

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.clip import _transform, load
from model.model import convert_weights, CLIP, IM2TEXT, VisualTransformer
from model.residual_flow_matching_module import ConditionalFlowNet
from model.seq_flow_matching_module import TokenFlowNet

from eval_utils import (
    evaluate_imgnet_retrieval,
    evaluate_coco,
    evaluate_fashion,
    evaluate_cirr,
    evaluate_cirr_fm,
    evaluate_cirr_test,
    evaluate_fashion_fm,
)

from data import CsvCOCO, FashionIQ, CIRR, ImageList
from params import parse_args, get_project_root
from logger import setup_primary_logging, setup_worker_logging
from utils import is_master, convert_models_to_fp32, TargetPad


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def maybe_strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def setup_log_save(args):
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"{name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    if args.dp:
        args.batch_size *= args.world_size

    if args.gpu is not None and torch.cuda.is_available():
        logging.info(f"Use GPU: {args.gpu} for evaluation")
        torch.cuda.set_device(args.gpu)


def build_backbone_and_preprocess(args):
    """
    Build CLIP backbone and preprocess.
    """
    if args.openai_pretrained:
        model, _, preprocess_val = load(args.model, jit=False)
    else:
        model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
        logging.info(f"Loading model from {model_config_file}")
        assert os.path.exists(model_config_file), f"Missing model config: {model_config_file}"

        with open(model_config_file, "r") as f:
            model_info = json.load(f)

        if args.use_prefix:
            model_info["vocab_size"] += 1
            model_info["use_prefix"] = True

        model = CLIP(**model_info)
        convert_weights(model)
        preprocess_val = _transform(model.visual.input_resolution, is_train=False)

    return model, preprocess_val


def build_img2text(model, args):
    """
    Build img2text mapper.
    """
    output_dim = model.token_embedding.weight.shape[1]
    embed_dim = getattr(model, "embed_dim", output_dim)

    try:
        img2text = IM2TEXT(
            embed_dim=embed_dim,
            middle_dim=args.middle_dim,
            output_dim=output_dim,
            n_layer=args.n_layer,
        )
    except Exception as e:
        logging.warning(f"Falling back to simplified IM2TEXT init due to: {e}")
        img2text = IM2TEXT(
            embed_dim=embed_dim,
            output_dim=output_dim,
            is_normalize=args.normalize_output,
            is_mlp=args.use_mlp,
            n_layer=args.n_layer,
        )
    return img2text


def build_flow_net(model, args):
    """
    Build flow net for compatibility with Flow Matching checkpoints.
    """
    if getattr(args, "loss_type", "global") == "global":
        flow_embed_dim = getattr(model, "embed_dim", 1024)
        flow_net = ConditionalFlowNet(
            dim=flow_embed_dim,
            time_dim=args.flow_time_dim,
            hidden_dim=args.flow_hidden_dim,
            use_delta=getattr(args, "global_flow_use_delta", True),
            use_condition=getattr(args, "global_flow_conditioning", "enabled") == "enabled",
            use_cond_gate=getattr(args, "global_flow_use_cond_gate", True),
        )
    elif getattr(args, "loss_type", "global") == "sequence":
        if not isinstance(model.visual, VisualTransformer):
            raise ValueError(
                "Sequence flow matching evaluation requires a ViT-based CLIP visual encoder."
            )
        flow_net = TokenFlowNet(
            text_dim=model.transformer_width,
            vis_dim=model.visual.conv1.out_channels,
            model_dim=args.seq_flow_model_dim,
            depth=args.seq_flow_depth,
            num_heads=args.seq_flow_heads,
            time_dim=args.flow_time_dim,
            num_vis_queries=args.seq_flow_num_vis_queries,
            dropout=args.seq_flow_dropout,
            predict_residual=args.seq_flow_predict_residual,
        )
    else:
        raise ValueError(f"Unsupported loss_type: {args.loss_type}")
    return flow_net


def get_device(args):
    if torch.cuda.is_available() and args.gpu is not None:
        return torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def prepare_modules_for_eval(model, img2text, flow_net, args):
    """
    Move modules to device and set eval mode.
    For evaluation, DDP is usually unnecessary unless your eval_utils explicitly depend on it.
    """
    device = get_device(args)

    if args.precision in ["amp", "fp32"] or device.type == "cpu":
        convert_models_to_fp32(model)
        convert_models_to_fp32(img2text)
        convert_models_to_fp32(flow_net)

    model.to(device)
    img2text.to(device)
    flow_net.to(device)

    if device.type == "cpu":
        model.float()
        img2text.float()
        flow_net.float()
        logging.warning("Using CPU, this will be slow.")
    else:
        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)
            convert_weights(flow_net)

        # eval stage usually does not need DDP wrapping for frozen modules
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
            img2text = torch.nn.DataParallel(img2text, device_ids=args.multigpu)
            flow_net = torch.nn.DataParallel(flow_net, device_ids=args.multigpu)

    model.eval()
    img2text.eval()
    flow_net.eval()

    return model, img2text, flow_net


def resolve_resume_path(args):
    if args.resume == "auto":
        args.resume = None
        if os.path.isdir(args.checkpoint_path):
            checkpoint_list = [
                ckpt for ckpt in os.listdir(args.checkpoint_path)
                if ckpt.endswith(".pt")
            ]

            numbered_ckpts = []
            latest_ckpt = None

            for ckpt in checkpoint_list:
                if ckpt == "epoch_latest.pt":
                    latest_ckpt = os.path.join(args.checkpoint_path, ckpt)
                elif ckpt.startswith("epoch_"):
                    try:
                        ep = int(ckpt.split("_")[1].split(".")[0])
                        numbered_ckpts.append((ep, os.path.join(args.checkpoint_path, ckpt)))
                    except Exception:
                        pass

            if len(numbered_ckpts) > 0:
                numbered_ckpts.sort(key=lambda x: x[0])
                args.resume = numbered_ckpts[-1][1]
            elif latest_ckpt is not None and os.path.isfile(latest_ckpt):
                args.resume = latest_ckpt

    return args.resume


def load_checkpoint_for_eval(model, img2text, flow_net, args):
    """
    Compatible with:
      1) old ckpt: state_dict + state_dict_img2text
      2) new ckpt: + state_dict_flow_net
    """
    device = get_device(args)
    resume_path = resolve_resume_path(args)

    assert resume_path is not None, "args.resume is None. Please specify --resume or use --resume auto."

    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"No checkpoint found at '{resume_path}'")

    logging.info(f"=> loading checkpoint '{resume_path}'")
    checkpoint = torch.load(resume_path, map_location=device)

    if "state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing 'state_dict' for model")

    sd_model = maybe_strip_module_prefix(checkpoint["state_dict"])
    unwrap_model(model).load_state_dict(sd_model, strict=True)

    if "state_dict_img2text" in checkpoint:
        sd_img2text = maybe_strip_module_prefix(checkpoint["state_dict_img2text"])
        unwrap_model(img2text).load_state_dict(sd_img2text, strict=True)
    else:
        logging.warning("Checkpoint does not contain 'state_dict_img2text'. Using current img2text init.")

    if "state_dict_flow_net" in checkpoint:
        sd_flow = maybe_strip_module_prefix(checkpoint["state_dict_flow_net"])
        unwrap_model(flow_net).load_state_dict(sd_flow, strict=True)
        logging.info("Loaded flow_net weights from checkpoint.")
    else:
        logging.warning(
            "Checkpoint does not contain 'state_dict_flow_net'. "
            "This is fine if you are evaluating old img2text-based models, "
            "but flow-based evaluation logic will not use trained flow weights."
        )

    logging.info(f"=> loaded checkpoint '{resume_path}' (epoch {checkpoint.get('epoch', 'N/A')})")
    return model, img2text, flow_net


def load_model(args):
    model, preprocess_val = build_backbone_and_preprocess(args)
    img2text = build_img2text(model, args)
    flow_net = build_flow_net(model, args)

    model, img2text, flow_net = prepare_modules_for_eval(model, img2text, flow_net, args)
    model, img2text, flow_net = load_checkpoint_for_eval(model, img2text, flow_net, args)

    return model, img2text, flow_net, preprocess_val


def main_worker(gpu, ngpus_per_node, log_queue, args):
    args.gpu = gpu
    args.rank = gpu if gpu is not None else 0
    setup_worker_logging(args.rank, log_queue, args.log_level)

    setup_log_save(args)

    model, img2text, flow_net, preprocess_val = load_model(args)

    cudnn.benchmark = True
    cudnn.deterministic = False

    root_project = os.path.join(get_project_root(), 'data')

    # optional target padding
    if args.target_pad:
        trans_tmp = preprocess_val.transforms
        trans_tmp = [TargetPad(1.25)] + trans_tmp
        preprocess_val = T.Compose(trans_tmp)

    # --------------------------------------------------
    # Evaluation
    # NOTE:
    # Current eval_utils signatures still use (model, img2text, args, ...)
    # If you want true flow-based inference at test time,
    # you must also modify eval_utils.py to pass flow_net into the pipeline.
    # --------------------------------------------------
    if args.eval_mode == 'coco':
        trans_val = preprocess_val.transforms
        n_px = trans_val[1].size
        trans_val = [T.Resize(n_px, interpolation=Image.BICUBIC)] + trans_val[2:]
        preprocess_val_region = T.Compose(trans_val)

        source_dataset = CsvCOCO(
            transforms=preprocess_val,
            transforms_region=preprocess_val_region,
            root=root_project
        )
        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
        evaluate_coco(model, img2text, args, source_dataloader)

    elif args.eval_mode == 'cirr':
        source_dataset = CIRR(transforms=preprocess_val, root=root_project)
        target_dataset = CIRR(transforms=preprocess_val, root=root_project, mode='imgs')

        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
        evaluate_cirr_fm(
            model,
            img2text,
            args,
            source_dataloader,
            target_dataloader,
            flow_net=flow_net,
        )

    elif args.eval_mode == 'cirr_test':
        source_dataset = CIRR(transforms=preprocess_val, root=root_project, test=True)
        target_dataset = CIRR(transforms=preprocess_val, root=root_project, mode='imgs', test=True)

        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )

        results = evaluate_cirr_test(model, img2text, args, source_dataloader, target_dataloader)
        os.makedirs("res_cirr", exist_ok=True)
        for key, value in results.items():
            with open(os.path.join("res_cirr", key + ".json"), "w") as f:
                json.dump(value, f)

    elif args.eval_mode == 'fashion':
        assert args.source_data in ['dress', 'shirt', 'toptee']

        source_dataset = FashionIQ(
            cloth=args.source_data,
            transforms=preprocess_val,
            root=root_project,
            is_return_target_path=True
        )
        target_dataset = FashionIQ(
            cloth=args.source_data,
            transforms=preprocess_val,
            root=root_project,
            mode='imgs'
        )

        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
        evaluate_fashion_fm(model, img2text, args, source_dataloader, target_dataloader, flow_net=flow_net)

    elif args.eval_mode == 'imgnet':
        domains = ['cartoon', 'origami', 'toy', 'sculpture']
        prompt = [f"a {domain} of *" for domain in domains]

        source_path = os.path.join(root_project, "imgnet", "imgnet_real_query.txt")
        target_path = os.path.join(root_project, "imgnet", "imgnet_targets.txt")

        source_dataset = ImageList(
            source_path,
            root=root_project,
            transforms=preprocess_val,
            is_labels=True
        )
        target_dataset = ImageList(
            target_path,
            root=root_project,
            transforms=preprocess_val,
            is_labels=True
        )

        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )

        evaluate_imgnet_retrieval(model, img2text, args, prompt, source_dataloader, target_dataloader)

    else:
        raise ValueError(f"Unsupported eval_mode: {args.eval_mode}")


def main():
    args = parse_args()

    # default flow args for compatibility
    if not hasattr(args, "flow_hidden_dim"):
        args.flow_hidden_dim = 4096
    if not hasattr(args, "flow_time_dim"):
        args.flow_time_dim = 128
    if not hasattr(args, "global_start_noise_std"):
        args.global_start_noise_std = 0.0

    args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)

    if args.name is None:
        args.name = (
            f"eval_"
            f"model={args.model}_"
            f"batchsize={args.batch_size}_"
            f"workers={args.workers}_"
            f"mode={args.eval_mode}"
        )
        if args.time_suffix:
            args.name += "_date=%Y-%m-%d-%H-%M-%S"
            args.name = strftime(args.name, gmtime())

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    os.makedirs(os.path.join(args.logs, args.name), exist_ok=True)

    assert args.precision in ['amp', 'fp16', 'fp32']

    args.ngpus_per_node = torch.cuda.device_count()
    args.wandb = False
    args.tensorboard = False

    args.tensorboard_path = ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints_eval_placeholder")

    # For auto resume, if you evaluate a training run, you should point args.resume
    # directly to that run's checkpoint, or override checkpoint_path externally.
    # Here we keep the variable for compatibility.
    torch.multiprocessing.set_start_method("spawn", force=True)

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)

    args.world_size = 1
    main_worker(args.gpu, None, log_queue, args)


if __name__ == "__main__":
    main()
