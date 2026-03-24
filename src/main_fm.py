# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
import sys
import json
import logging
from pathlib import Path
from time import gmtime, strftime

import torch
import wandb
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, "/home/sunyw/CIR/Pic2Word")

from third_party.open_clip.scheduler import cosine_lr
from model.clip import _transform, load
from model.model import convert_weights, CLIP, IM2TEXT, VisualTransformer
from trainer_fm import train, validate
from data import get_data
from params import parse_args
from logger import setup_primary_logging, setup_worker_logging
from utils import is_master, convert_models_to_fp32

# ===== Flow Matching =====
from model.residual_flow_matching_module import ConditionalFlowNet
from model.flow_matching_loss import FlowMatchingLoss
from model.seq_flow_matching_loss import TokenFlowMatchingLoss
from model.seq_flow_matching_module import TokenFlowNet


def unwrap_model(model):
    """Return the underlying model for DDP / DP wrapped modules."""
    return model.module if hasattr(model, "module") else model


def freeze_module(module):
    """Freeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def build_img2text(model, args):
    """
    Build img2text mapper.
    Keep this module in forward pass, but we will freeze it later.
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


def get_model_device(args):
    if torch.cuda.is_available() and args.gpu is not None:
        return torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def maybe_strip_module_prefix(state_dict):
    """Strip leading 'module.' if present."""
    if len(state_dict) == 0:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def save_checkpoint(epoch, args, model, img2text, flow_net, optimizer, scaler=None, extra=None):
    """
    Save checkpoint.
    Even if model / img2text are frozen, still save them for reproducibility and resume.
    """
    ckpt = {
        "epoch": epoch + 1,
        "name": args.name,
        "state_dict": unwrap_model(model).state_dict(),
        "state_dict_img2text": unwrap_model(img2text).state_dict(),
        "state_dict_flow_net": unwrap_model(flow_net).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    if extra is not None:
        ckpt.update(extra)
    return ckpt

def maybe_strip_module_prefix(state_dict):
    """Strip leading 'module.' if present."""
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def safe_load_module(module, state_dict, name="module", strict=True):
    """Load state_dict into module safely."""
    if state_dict is None:
        logging.warning(f"No state_dict provided for {name}.")
        return

    state_dict = maybe_strip_module_prefix(state_dict)
    msg = module.load_state_dict(state_dict, strict=strict)

    if strict:
        logging.info(f"Loaded {name} with strict={strict}.")
    else:
        missing = getattr(msg, "missing_keys", [])
        unexpected = getattr(msg, "unexpected_keys", [])
        logging.info(
            f"Loaded {name} with strict={strict}. "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if len(missing) > 0:
            logging.warning(f"{name} missing keys: {missing[:20]}")
        if len(unexpected) > 0:
            logging.warning(f"{name} unexpected keys: {unexpected[:20]}")

def main_worker(gpu, ngpus_per_node, log_queue, args):
    args.gpu = gpu
    args.rank = gpu if gpu is not None else 0
    setup_worker_logging(args.rank, log_queue, args.log_level)

    params_file = None

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
        logging.info(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)

    device = get_model_device(args)

    # --------------------------------------------------
    # 1. Backbone model
    # --------------------------------------------------
    if args.openai_pretrained:
        model, preprocess_train, preprocess_val = load(args.model, jit=False)
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
        preprocess_train = _transform(model.visual.input_resolution, is_train=True)
        preprocess_val = _transform(model.visual.input_resolution, is_train=False)

    # --------------------------------------------------
    # 2. img2text mapper
    # Important module, but frozen (no update)
    # --------------------------------------------------
    img2text = build_img2text(model, args)

    # --------------------------------------------------
    # 3. Flow net
    # Only this module is optimized
    # --------------------------------------------------
    if args.loss_type == "global":
        flow_embed_dim = getattr(model, "embed_dim", 1024)
        flow_net = ConditionalFlowNet(
            dim=flow_embed_dim,
            time_dim=args.flow_time_dim,
            hidden_dim=args.flow_hidden_dim,
            use_delta=args.global_flow_use_delta,
            use_condition=args.global_flow_conditioning == "enabled",
            use_cond_gate=args.global_flow_use_cond_gate,
        )
    elif args.loss_type == "sequence":
        if not isinstance(model.visual, VisualTransformer):
            raise ValueError(
                "Sequence flow matching requires a ViT-based CLIP model so that visual token sequences are available."
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

    # --------------------------------------------------
    # 4. Freeze modules that should not be updated
    # --------------------------------------------------
    freeze_module(model)
    freeze_module(img2text)
    # flow_net stays trainable

    # --------------------------------------------------
    # 5. Precision / device
    # --------------------------------------------------
    if args.precision in ["amp", "fp32"] or device.type == "cpu":
        convert_models_to_fp32(model)
        convert_models_to_fp32(img2text)
        convert_models_to_fp32(flow_net)

    model.to(device)
    img2text.to(device)
    flow_net.to(device)

    if device.type == "cpu":
        logging.warning("Using CPU, this will be slow.")
    else:
        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)
            convert_weights(flow_net)

        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if args.distributed:
            # Only wrap trainable module with DDP
            flow_net = torch.nn.parallel.DistributedDataParallel(
                flow_net,
                device_ids=[args.gpu],
                find_unused_parameters=False,
            )

        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
            img2text = torch.nn.DataParallel(img2text, device_ids=args.multigpu)
            flow_net = torch.nn.DataParallel(flow_net, device_ids=args.multigpu)

    # --------------------------------------------------
    # 6. Data
    # --------------------------------------------------
    data = get_data(args, (preprocess_train, preprocess_val))

    # --------------------------------------------------
    # 7. Criterion
    # --------------------------------------------------
    if args.loss_type == "global":
        criterion = FlowMatchingLoss(
            flow_net=flow_net,
            num_steps=args.flow_num_steps,
            lambda_fm=args.lambda_fm,
            lambda_end=args.lambda_end,
            lambda_ret=args.lambda_ret,
            # lambda_mid=args.lambda_mid,
            temperature=args.flow_temperature,
            normalize=True,
        ).to(device)
    elif args.loss_type == "sequence":
        criterion = TokenFlowMatchingLoss(
            lambda_fm=args.lambda_fm,
            lambda_end=args.lambda_end,
            lambda_tok=args.lambda_tok,
            lambda_keep=args.lambda_keep,
            lambda_direct_end=args.lambda_direct_end,
            keep_threshold=args.flow_keep_threshold,
        ).to(device)
    else:
        raise ValueError(f"Unsupported loss_type: {args.loss_type}")

    # --------------------------------------------------
    # 8. Optimizer / Scheduler
    # Only optimize flow_net
    # --------------------------------------------------
    exclude = lambda n: ("bn" in n) or ("ln" in n) or ("bias" in n)
    include = lambda n: not exclude(n)

    named_parameters = list(unwrap_model(flow_net).named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    scaler = GradScaler() if args.precision == "amp" else None

    # --------------------------------------------------
    # 9. Resume / Pretrained Init
    # Compatible with:
    #   (1) old ckpt: state_dict + state_dict_img2text + optimizer
    #   (2) new ckpt: + state_dict_flow_net (+ scaler)
    # --------------------------------------------------
    start_epoch = 0

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

    if args.resume is not None:
        if os.path.isfile(args.resume):
            logging.info(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)

            # -------------------------
            # 1) load backbone model
            # -------------------------
            if "state_dict" in checkpoint:
                safe_load_module(
                    unwrap_model(model),
                    checkpoint["state_dict"],
                    name="model",
                    strict=True,
                )
            else:
                logging.warning("Checkpoint does not contain 'state_dict' for model.")

            # -------------------------
            # 2) load img2text
            # -------------------------
            if "state_dict_img2text" in checkpoint:
                safe_load_module(
                    unwrap_model(img2text),
                    checkpoint["state_dict_img2text"],
                    name="img2text",
                    strict=True,
                )
            else:
                logging.warning("Checkpoint does not contain 'state_dict_img2text'.")

            # -------------------------
            # 3) load flow_net if exists
            #    If absent, treat checkpoint as pretrained init
            # -------------------------
            has_flow = "state_dict_flow_net" in checkpoint
            if has_flow:
                safe_load_module(
                    unwrap_model(flow_net),
                    checkpoint["state_dict_flow_net"],
                    name="flow_net",
                    strict=True,
                )

                # Only resume epoch when flow_net exists
                start_epoch = checkpoint.get("epoch", 0)

                # -------------------------
                # 4) optimizer
                # Only load optimizer if flow_net weights also exist
                # -------------------------
                if optimizer is not None and checkpoint.get("optimizer", None) is not None:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                        logging.info("Loaded optimizer state.")
                    except Exception as e:
                        logging.warning(f"Failed to load optimizer state: {e}")

                # -------------------------
                # 5) scaler
                # -------------------------
                if scaler is not None and checkpoint.get("scaler", None) is not None:
                    try:
                        scaler.load_state_dict(checkpoint["scaler"])
                        logging.info("Loaded GradScaler state.")
                    except Exception as e:
                        logging.warning(f"Failed to load GradScaler state: {e}")

                logging.info(
                    f"=> resumed training from checkpoint '{args.resume}' "
                    f"(epoch {start_epoch})"
                )
            else:
                # old checkpoint: only initialize model + img2text
                start_epoch = 0
                logging.warning(
                    "Checkpoint has no 'state_dict_flow_net'. "
                    "Will use it as pretrained initialization for model + img2text only. "
                    "flow_net remains randomly initialized, optimizer/scaler will not be resumed."
                )

        else:
            logging.info(f"=> no checkpoint found at '{args.resume}'")

    # --------------------------------------------------
    # 10. Logging
    # --------------------------------------------------
    args.save_logs = (
        args.logs is not None
        and args.logs != ""
        and args.logs.lower() != "none"
    ) and ((not args.distributed) or args.gpu == 0)

    writer = None
    if args.save_logs and args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        logging.debug("Starting wandb.")
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None and "val" in data:
            args.val_sz = data["val"].dataloader.num_samples

        wandb.init(
            project="flow-cir",
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )

        if args.debug:
            wandb.watch(unwrap_model(flow_net), log="all")

        if params_file is not None and os.path.isfile(params_file):
            wandb.save(params_file)

        logging.debug("Finished loading wandb.")

    # --------------------------------------------------
    # 11. Train
    # --------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f"Start epoch {epoch}")

        # keep frozen modules in eval mode
        unwrap_model(model).eval()
        unwrap_model(img2text).eval()
        unwrap_model(flow_net).train()

        train(
            model=model,
            img2text=img2text,
            flow_net=flow_net,
            criterion=criterion,
            data=data,
            epoch=epoch,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            args=args,
            writer=writer,
        )

        if (epoch + 1) % 1 == 0:
            validate(
                model=model,
                img2text=img2text,
                flow_net=flow_net,
                criterion=criterion,
                data=data,
                epoch=epoch + 1,
                args=args,
                writer=writer,
            )

            # restore training mode after validation
            unwrap_model(model).eval()
            unwrap_model(img2text).eval()
            unwrap_model(flow_net).train()

        if args.save_logs and (args.gpu == 0 or (not args.distributed)):
            ckpt = save_checkpoint(
                epoch=epoch,
                args=args,
                model=model,
                img2text=img2text,
                flow_net=flow_net,
                optimizer=optimizer,
                scaler=scaler,
            )

            if (epoch + 1) == args.epochs or (
                args.save_frequency > 0 and ((epoch + 1) % args.save_frequency) == 0
            ):
                torch.save(
                    ckpt,
                    os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"),
                )

            if args.save_most_recent:
                torch.save(
                    ckpt,
                    os.path.join(args.checkpoint_path, "epoch_latest.pt"),
                )

    if args.wandb and (args.gpu == 0 or (not args.distributed)):
        wandb.finish()

    if args.distributed:
        dist.destroy_process_group()


def main():
    
    args = parse_args()

    # --------------------------------------------------
    # Default flow args
    # --------------------------------------------------
    if not hasattr(args, "flow_hidden_dim"):
        args.flow_hidden_dim = 4096
    if not hasattr(args, "flow_time_dim"):
        args.flow_time_dim = 128
    if not hasattr(args, "flow_num_steps"):
        args.flow_num_steps = 4
    if not hasattr(args, "flow_temperature"):
        args.flow_temperature = 0.07
    if not hasattr(args, "lambda_fm"):
        args.lambda_fm = 1.0
    if not hasattr(args, "lambda_end"):
        args.lambda_end = 1.0
    if not hasattr(args, "lambda_ret"):
        args.lambda_ret = 0.05
    if not hasattr(args, "lambda_mid"):
        args.lambda_mid = 0.5
    if not hasattr(args, "global_start_noise_std"):
        args.global_start_noise_std = 0.0
    if not hasattr(args, "loss_type"):
        args.loss_type = "global"

    if args.name is None:
        args.name = (
            f"lr={args.lr}_"
            f"wd={args.wd}_"
            f"model={args.model}_"
            f"bs={args.batch_size}_"
            f"flow={args.loss_type}_"
            f"flowhd={args.flow_hidden_dim}_"
            f"steps={args.flow_num_steps}_"
            f"gnoise={args.global_start_noise_std}"
        )
        if args.time_suffix:
            args.name += "_date=%Y-%m-%d-%H-%M-%S"
            args.name = strftime(args.name, gmtime())

    if args.copy_codebase:
        import subprocess
        from shutil import copytree, ignore_patterns

        new_code_path = os.path.join(args.logs, args.name, "code")
        if os.path.exists(new_code_path):
            print(f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment.")
            return -1

        print(f"Copying codebase to {new_code_path}")
        current_code_path = os.path.realpath(__file__)
        for _ in range(3):
            current_code_path = os.path.dirname(current_code_path)

        copytree(
            current_code_path,
            new_code_path,
            ignore=ignore_patterns("log", "logs", "wandb"),
        )
        print("Done copying code.")

        os.environ["PYTHONPATH"] = (
            f"{os.environ.get('PYTHONPATH', '')}:"
            f"{os.path.join(new_code_path, 'src')}"
        )

        argv = sys.argv[:]
        if "--copy-codebase" in argv:
            argv.remove("--copy-codebase")
        argv.extend(["--name", args.name])
        command = [sys.executable] + argv
        print("Executing command:", " ".join(command))
        subprocess.check_call(command)
        return 1

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    if os.path.exists(args.log_path) and args.resume is None:
        print("Error. Experiment already exists. Use --name to specify a new experiment.")
        return -1

    assert args.precision in ["amp", "fp16", "fp32"]

    args.ngpus_per_node = torch.cuda.device_count()
    args.wandb = "wandb" in args.report_to or "all" in args.report_to
    args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to

    args.tensorboard_path = (
        os.path.join(args.logs, args.name, "tensorboard")
        if args.tensorboard else ""
    )
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")

    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    torch.multiprocessing.set_start_method("spawn", force=True)

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)

    args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)

    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, log_queue, args),
        )
    else:
        if args.dp:
            args.gpu = args.multigpu[0]
            args.world_size = len(args.multigpu)
        else:
            args.world_size = 1
        main_worker(args.gpu, None, log_queue, args)


if __name__ == "__main__":
    main()
