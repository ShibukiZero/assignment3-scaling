from __future__ import annotations

import hashlib
import logging
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from cs336_scaling.api_contract import TrainingConfig
from cs336_scaling.model import BasicsTransformerLM
from cs336_scaling.token_dataset import TokenizedDataset
from cs336_scaling.training_budget import TrainingPlan, build_training_plan

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class TrainingRunResult:
    loss: float
    steps_completed: int
    training_plan: TrainingPlan


def _stable_seed(config: TrainingConfig) -> int:
    payload = (
        f"{config.api_key}|{config.d_model}|{config.num_layers}|{config.num_heads}|"
        f"{config.batch_size}|{config.learning_rate}|{config.train_flops}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _normalize_mixed_precision(device: torch.device, mixed_precision: str) -> str:
    normalized = mixed_precision.strip().lower()
    if normalized not in {"off", "bf16", "fp16"}:
        raise ValueError(f"Unsupported mixed precision mode: {mixed_precision}")
    if device.type != "cuda":
        return "off"
    return normalized


def _resolve_autocast_dtype(mixed_precision: str) -> torch.dtype | None:
    if mixed_precision == "off":
        return None
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported mixed precision mode: {mixed_precision}")


class TrainingRunner:
    def __init__(
        self,
        *,
        train_data_meta_path: str | Path,
        vocab_size: int,
        context_length: int,
        device: str = "cpu",
        mixed_precision: str = "off",
        activation_checkpointing: bool = False,
        preload_dataset: bool = True,
        dataset: TokenizedDataset | None = None,
        max_steps_cap: int | None = None,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
        residual_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
    ) -> None:
        self.train_data_meta_path = Path(train_data_meta_path).expanduser()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.device = torch.device(device)
        self.mixed_precision = _normalize_mixed_precision(self.device, mixed_precision)
        self.autocast_dtype = _resolve_autocast_dtype(self.mixed_precision)
        self.activation_checkpointing = activation_checkpointing
        self.preload_dataset = preload_dataset
        self.max_steps_cap = max_steps_cap
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.residual_pdrop = residual_pdrop
        self.attn_pdrop = attn_pdrop
        self.dataset = dataset
        if self.dataset is None and self.preload_dataset:
            self.dataset = TokenizedDataset.from_meta(self.train_data_meta_path)

    def _build_model(self, config: TrainingConfig) -> BasicsTransformerLM:
        model = BasicsTransformerLM(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=4 * config.d_model,
            attn_pdrop=self.attn_pdrop,
            residual_pdrop=self.residual_pdrop,
            activation_checkpointing=self.activation_checkpointing,
        )
        return model.to(self.device)

    def _capped_plan(self, config: TrainingConfig) -> TrainingPlan:
        plan = build_training_plan(config=config, context_length=self.context_length)
        if self.max_steps_cap is None:
            return plan
        capped_steps = max(1, min(plan.max_steps, self.max_steps_cap))
        return replace(
            plan,
            max_steps=capped_steps,
            effective_train_tokens=capped_steps * plan.tokens_per_step,
        )

    def _sample_batch(
        self,
        dataset: TokenizedDataset,
        *,
        batch_size: int,
        context_length: int,
        rng: random.Random,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_length = context_length + 1
        max_start = dataset.num_tokens - block_length
        starts = [rng.randint(0, max_start) for _ in range(batch_size)]

        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        for start in starts:
            block = dataset.get_token_block(start=start, length=block_length)
            inputs.append(block[:-1])
            targets.append(block[1:])

        x = torch.tensor(inputs, dtype=torch.long, device=self.device)
        y = torch.tensor(targets, dtype=torch.long, device=self.device)
        return x, y

    def _make_scheduler(self, optimizer: torch.optim.Optimizer, max_steps: int):
        if max_steps <= 1:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        def lr_lambda(step_index: int) -> float:
            progress = min(step_index / max_steps, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return 0.1 + 0.9 * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _autocast_context(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def run(self, config: TrainingConfig) -> TrainingRunResult:
        dataset = self.dataset or TokenizedDataset.from_meta(self.train_data_meta_path)
        if dataset.num_tokens < self.context_length + 1:
            raise ValueError("Tokenized corpus must contain at least context_length + 1 tokens.")

        plan = self._capped_plan(config)
        logger.info(
            "training plan for api_key=%s device=%s d_model=%s layers=%s heads=%s batch=%s lr=%s flops=%s max_steps=%s train_tokens=%.2f effective_train_tokens=%s tokens_per_step=%s mixed_precision=%s activation_checkpointing=%s",
            config.api_key,
            self.device,
            config.d_model,
            config.num_layers,
            config.num_heads,
            config.batch_size,
            config.learning_rate,
            config.train_flops,
            plan.max_steps,
            plan.train_tokens,
            plan.effective_train_tokens,
            plan.tokens_per_step,
            self.mixed_precision,
            self.activation_checkpointing,
        )
        seed = _stable_seed(config)
        rng = random.Random(seed)
        fork_devices = []
        if self.device.type == "cuda":
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            fork_devices = [device_index]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)

            model = self._build_model(config)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=self.weight_decay,
            )
            scheduler = self._make_scheduler(optimizer, plan.max_steps)
            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=self.device.type == "cuda" and self.mixed_precision == "fp16",
            )

            final_loss = 0.0
            model.train()
            for _ in range(plan.max_steps):
                x, y = self._sample_batch(
                    dataset,
                    batch_size=config.batch_size,
                    context_length=self.context_length,
                    rng=rng,
                )
                optimizer.zero_grad(set_to_none=True)

                with self._autocast_context():
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                    optimizer.step()

                scheduler.step()
                final_loss = float(loss.detach().item())

        result = TrainingRunResult(
            loss=final_loss,
            steps_completed=plan.max_steps,
            training_plan=plan,
        )
        logger.info(
            "training completed for api_key=%s device=%s steps_completed=%s final_loss=%s",
            config.api_key,
            self.device,
            result.steps_completed,
            result.loss,
        )
        return result
