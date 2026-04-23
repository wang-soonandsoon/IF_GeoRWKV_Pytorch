from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import yaml

import utility
from model.factory import get_model_spec
from setting.dataLoader import get_loader
from setting.options import opt
from setting.utils import compute_accuracy, create_folder, random_seed_setting

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

seed = int(getattr(opt, "seed", 6))
deterministic = bool(int(getattr(opt, "deterministic", 0)))
allow_tf32 = bool(int(getattr(opt, "allow_tf32", 1)))

random_seed_setting(seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

torch.backends.cudnn.enabled = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

MODEL_NAME = opt.model
save_path = create_folder(os.path.join(opt.save_path, MODEL_NAME, opt.dataset))
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

log_dir = os.path.join(save_path, "log")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(save_path, "weight"), exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, f"{opt.dataset}{current_time}log.log"),
    format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
    level=logging.INFO,
    filemode="a",
    datefmt="%Y-%m-%d %I:%M:%S %p",
)

logging.info("********************start train!********************")
logging.info(
    f"Config--epoch:{opt.epoch}; lr:{opt.lr}; batch_size:{opt.batchsize}; "
    f"weight_decay:{getattr(opt, 'weight_decay', 0.0)}; wd_mode:{getattr(opt, 'wd_mode', 'all')}; "
    f"adam_eps:{getattr(opt, 'adam_eps', 1e-8)}; "
    f"adam_betas:({getattr(opt, 'adam_beta1', 0.9)},{getattr(opt, 'adam_beta2', 0.999)}); "
    f"label_smoothing:{getattr(opt, 'label_smoothing', 0.0)}; "
    f"seed:{seed}; deterministic:{int(deterministic)}; allow_tf32:{int(allow_tf32)}; "
    f"amp:{int(getattr(opt, 'amp', 0))}; amp_dtype:{getattr(opt, 'amp_dtype', 'bf16')}; "
    f"compile_model:{int(getattr(opt, 'compile_model', 0))}; cache_pca:{int(getattr(opt, 'cache_pca', 1))}; "
    f"persistent_workers:{int(getattr(opt, 'persistent_workers', 1))}; prefetch_factor:{getattr(opt, 'prefetch_factor', 2)}; "
    f"init_weights:{getattr(opt, 'init_weights', '') or '-'}; init_weights_strict:{int(getattr(opt, 'init_weights_strict', 1))};"
)

shape_log_path = os.path.join(log_dir, f"{opt.dataset}_{current_time}_shapes.yaml")
train_loader, test_loader, trntst_loader, all_loader, train_num, val_num, trntst_num = get_loader(
    dataset=opt.dataset,
    batchsize=opt.batchsize,
    num_workers=opt.num_work,
    useval=opt.useval,
    pin_memory=True,
    data_root=opt.data_root,
    shape_log_path=shape_log_path,
    cache_pca=bool(int(getattr(opt, "cache_pca", 1))),
    return_full_hsi=bool(int(getattr(opt, "return_full_hsi", 0))),
    persistent_workers=bool(int(getattr(opt, "persistent_workers", 1))),
    prefetch_factor=int(getattr(opt, "prefetch_factor", 2)),
)

logging.info(
    f"Loading data, including {train_num} training images and {val_num} validation images and {trntst_num} train_test images"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda" and bool(int(getattr(opt, "amp", 0)))
amp_dtype_str = str(getattr(opt, "amp_dtype", "bf16")).strip().lower()
if amp_dtype_str == "bf16" and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
    amp_dtype_str = "fp16"
amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
use_grad_scaler = use_amp and amp_dtype == torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)


def autocast_context():
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


model_spec = get_model_spec(opt.model)
model = model_spec.create(opt.dataset).to(device)

init_weights = str(getattr(opt, "init_weights", "") or "").strip()
if init_weights:
    if not os.path.isfile(init_weights):
        raise FileNotFoundError(f"init_weights not found: {init_weights}")
    strict = bool(int(getattr(opt, "init_weights_strict", 1)))
    state = torch.load(init_weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    load_result = model.load_state_dict(state, strict=strict)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    print(
        f"[init_weights] loaded={init_weights} strict={int(strict)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    logging.info(
        f"[init_weights] loaded={init_weights} strict={int(strict)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        logging.info(f"[init_weights] missing_keys(head): {missing[:20]}")
    if unexpected:
        logging.info(f"[init_weights] unexpected_keys(head): {unexpected[:20]}")

run_model = model
if bool(int(getattr(opt, "compile_model", 0))):
    try:
        run_model = torch.compile(model)
        logging.info("[compile] torch.compile enabled")
    except Exception as exc:
        logging.info(f"[compile] failed, falling back to eager mode: {exc}")
        run_model = model

with (Path(__file__).resolve().parent / "dataset_info.yaml").open("r", encoding="utf-8") as f:
    dataset_info = yaml.safe_load(f)
num_classes = int(getattr(model, "out_features", 0) or dataset_info[opt.dataset]["num_classes"])

weight_decay = float(getattr(opt, "weight_decay", 0.0) or 0.0)
wd_mode = str(getattr(opt, "wd_mode", "all") or "all").strip().lower()
adam_eps = float(getattr(opt, "adam_eps", 1e-8) or 1e-8)
adam_beta1 = float(getattr(opt, "adam_beta1", 0.9) or 0.9)
adam_beta2 = float(getattr(opt, "adam_beta2", 0.999) or 0.999)

if wd_mode not in {"all", "matrix_only"}:
    raise ValueError(f"Unsupported wd_mode={wd_mode!r}, choose from all/matrix_only")

if wd_mode == "all":
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=weight_decay,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )
else:
    decay_params = []
    no_decay_params = []
    no_decay_tokens = ("time_", "gate_", "norm", "layernorm", "ln_")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower_name = name.lower()
        is_bias = lower_name.endswith(".bias")
        is_matrix = param.ndim >= 2
        has_no_decay_token = any(tok in lower_name for tok in no_decay_tokens)
        if is_matrix and not is_bias and not has_no_decay_token:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=opt.lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )
    logging.info(
        f"[optimizer] wd_mode=matrix_only decay_params={len(decay_params)} no_decay_params={len(no_decay_params)}"
    )

criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(getattr(opt, "label_smoothing", 0.0))).to(device)
grad_clip_norm = float(getattr(opt, "grad_clip_norm", 0.0) or 0.0)
ema_decay = float(getattr(opt, "ema_decay", 0.0) or 0.0)


class _EMA:
    def __init__(self, base_model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if not param.requires_grad:
                    continue
                self.shadow[name] = param.detach().clone()

    def update(self, base_model: torch.nn.Module) -> None:
        d = self.decay
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name not in self.shadow:
                    continue
                self.shadow[name].mul_(d).add_(param.detach(), alpha=(1.0 - d))

    def apply(self, base_model: torch.nn.Module) -> None:
        self.backup = {}
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name not in self.shadow:
                    continue
                self.backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])

    def restore(self, base_model: torch.nn.Module) -> None:
        if not self.backup:
            return
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name not in self.backup:
                    continue
                param.copy_(self.backup[name])
        self.backup = {}


ema = _EMA(model, ema_decay) if ema_decay > 0.0 else None


@contextmanager
def _eval_weights_context(base_model: torch.nn.Module):
    if ema is None:
        yield
        return
    ema.apply(base_model)
    try:
        yield
    finally:
        ema.restore(base_model)


def _forward_logits_eval(eval_model: torch.nn.Module, batch, device_t: torch.device):
    with autocast_context():
        return model_spec.forward_logits(eval_model, batch, device_t)


best_acc = opt.best_acc
best_epoch = opt.best_epoch


def train_one_epoch(train_loader, run_model, raw_model, optimizer, epoch, save_path):
    run_model.train()
    loss_all = 0.0
    extra_all = 0.0
    iteration = len(train_loader)
    acc = 0.0
    num = 0

    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        gt = model_spec.get_targets(batch, device)
        with autocast_context():
            outputs, extra_loss = model_spec.forward_train(run_model, batch, device)
            gt_loss = criterion(outputs, gt)
            loss = gt_loss + extra_loss

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        if ema is not None:
            ema.update(raw_model)

        loss_all += float(loss.detach().cpu())
        extra_all += float(extra_loss.detach().cpu())
        acc += compute_accuracy(outputs.detach().float(), gt) * len(gt)
        num += len(gt)

    loss_avg = loss_all / max(iteration, 1)
    extra_avg = extra_all / max(iteration, 1)
    acc_avg = acc / max(num, 1)
    logging.info(
        f"Epoch [{epoch:03d}/{opt.epoch:03d}], Loss_train_avg: {loss_avg:.4f}, "
        f"Extra_loss_avg: {extra_avg:.6f}, acc_avg:{acc_avg:.4f}"
    )

    if epoch == opt.epoch or epoch == opt.epoch // 2:
        torch.save(
            optimizer.state_dict(),
            os.path.join(save_path, "weight", f"{current_time}_{MODEL_NAME}_{opt.dataset}_optimizerEpoch{epoch}.pth"),
        )
        torch.save(
            raw_model.state_dict(),
            os.path.join(save_path, "weight", f"{current_time}_{MODEL_NAME}_{opt.dataset}_Net_epoch_{epoch}.pth"),
        )


def evaluate(val_loader, eval_model, raw_model, epoch, save_path):
    global best_acc, best_epoch
    eval_model.eval()
    with _eval_weights_context(raw_model):
        oa, aa, kappa, acc = utility.createAutoReport(
            net=eval_model,
            data=val_loader,
            device=device,
            num_classes=num_classes,
            forward_logits_fn=_forward_logits_eval,
        )
    if oa > best_acc:
        best_acc, best_epoch = oa, epoch
        if epoch >= 1:
            torch.save(
                optimizer.state_dict(),
                os.path.join(
                    save_path,
                    "weight",
                    f"{current_time}_{best_acc}_{MODEL_NAME}_{opt.dataset}_optimizerEpoch{epoch}.pth",
                ),
            )
            torch.save(
                raw_model.state_dict(),
                os.path.join(
                    save_path,
                    "weight",
                    f"{current_time}_{best_acc}_{MODEL_NAME}_{opt.dataset}_Net_epoch_{epoch}.pth",
                ),
            )
    print(f"Epoch [{epoch:03d}/{opt.epoch:03d}] best_acc={best_acc:.4f}, Best_epoch:{best_epoch:03d}")
    logging.info(f"Best_acc:{best_acc:.4f},Best_epoch:{best_epoch:03d}")


if __name__ == "__main__":
    print("Start train...")
    time_begin = time.time()

    if opt.dry_run:
        run_model.train()
        batch = next(iter(train_loader))
        optimizer.zero_grad(set_to_none=True)
        gt = model_spec.get_targets(batch, device)
        with autocast_context():
            outputs, extra_loss = model_spec.forward_train(run_model, batch, device)
            loss = criterion(outputs, gt) + extra_loss
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        print(
            f"[dry_run] dataset={opt.dataset} loss={loss.item():.4f} extra={float(extra_loss.detach().cpu()):.6f} "
            f"outputs={tuple(outputs.shape)} dtype={outputs.dtype}"
        )
        raise SystemExit(0)

    for epoch in range(opt.start_epoch, opt.epoch + 1):
        print(f"Epoch [{epoch:03d}/{opt.epoch:03d}] training...")
        train_one_epoch(train_loader, run_model, model, optimizer, epoch, save_path)

        eval_interval = int(getattr(opt, "eval_interval", 1))
        if eval_interval <= 0:
            do_eval = epoch == opt.epoch
        else:
            do_eval = (epoch % eval_interval == 0) or (epoch == opt.epoch)

        if do_eval:
            print(f"Epoch [{epoch:03d}/{opt.epoch:03d}] evaluating...")
            evaluate(test_loader, model, model, epoch, save_path)
        else:
            print(f"Epoch [{epoch:03d}/{opt.epoch:03d}] skip eval (eval_interval={eval_interval})")

        time_epoch = time.time()
        print(f"Time out:{time_epoch - time_begin:.2f}s\n")
        logging.info(f"Time out:{time_epoch - time_begin:.2f}s\n")
