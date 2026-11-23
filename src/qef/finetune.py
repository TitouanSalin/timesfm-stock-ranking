"""Building blocks for fine-tuning TimesFM 3 on daily stock returns.

The package only documents inference, and its decode method is wrapped in
torch.no_grad. decode_with_grad calls the same undecorated function, so training
uses exactly the prediction path. The resulting weights fall under the TimesFM
non-commercial license and must not be distributed.
"""

from __future__ import annotations

import inspect
import math
import time

import numpy as np
import torch

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN = 4
CONTEXT = 512

# The package always decodes one full output patch, and the number of horizon patches
# changes the attention pattern, so training must request the same horizon.
DECODE_HORIZON = 64


def check_weights(model) -> None:
    """Stop at once if the loaded weights are invalid, as happened on one faulty GPU node.

    Forecasts from such weights raise no error, so they could otherwise pass unnoticed.
    """
    invalid = [name for name, value in model.state_dict().items() if value.is_floating_point() and not torch.isfinite(value).all()]
    if invalid:
        raise SystemExit(f"{len(invalid)} weight tensors hold invalid values, the first being {invalid[0]}. Exclude this node.")


def decode_with_grad(model, contexts: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Forecast quantiles of shape (batch, n_steps, 9) from contexts of shape (batch, length)."""
    decode = inspect.unwrap(type(model).decode)
    logits = decode(model, target=contexts[:, None, :], horizon=DECODE_HORIZON)
    return logits[:, 0, :n_steps, :]


def set_trainable(model, scope: str) -> int:
    """Choose what is trained: 'full', 'last_<k>' layers plus the head, or 'head'. Returns the parameter count."""
    model.requires_grad_(False)
    if scope == "full":
        model.requires_grad_(True)
    else:
        model.output_head.requires_grad_(True)
        if scope.startswith("last_"):
            for layer in model.transformer_stack.layers[-int(scope.split("_")[1]) :]:
                layer.requires_grad_(True)
        elif scope != "head":
            raise ValueError(f"Unknown scope: {scope}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pinball_loss(quantiles: torch.Tensor, target: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Mean quantile loss. Dividing by each window's scale keeps volatile stocks from dominating."""
    levels = torch.tensor(QUANTILES, dtype=quantiles.dtype, device=quantiles.device)
    errors = target[..., None] - quantiles
    loss = torch.maximum(levels * errors, (levels - 1) * errors).mean(dim=(1, 2))
    if scale is not None:
        loss = loss / scale
    return loss.mean()


def correlation_loss(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Minus the correlation between scores and outcomes across the stocks of one date."""
    score = score - score.mean()
    target = target - target.mean()
    return -(score * target).sum() / (score.norm() * target.norm() + 1e-8)


def five_day_score(quantiles: torch.Tensor, skip: int, horizon: int) -> torch.Tensor:
    """Expected return over the target period, from the median of daily return forecasts."""
    return quantiles[:, skip : skip + horizon, MEDIAN].sum(dim=1)


def eligible_rows(mask: np.ndarray, history: np.ndarray, context: int, first: int, last: int) -> np.ndarray:
    """(date, stock) positions in [first, last) that are masked and have a full context."""
    dates, stocks = np.nonzero(mask[first:last] & (history[first:last] >= context))
    return np.column_stack([dates + first, stocks])


def make_batch(series: np.ndarray, rows: np.ndarray, context: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Contexts ending at t, and the returns of t+1 to t+n_steps, for each (t, stock) row."""
    dates, stocks = rows[:, 0], rows[:, 1]
    past = dates[:, None] + np.arange(-context + 1, 1)
    future = dates[:, None] + np.arange(1, n_steps + 1)
    return series[past, stocks[:, None]], series[future, stocks[:, None]]


def to_device(array: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32)).to(device)


def batch_loss(model, series, rows, n_steps, device, context: int = CONTEXT) -> torch.Tensor:
    contexts, future = make_batch(series, rows, context, n_steps)
    contexts, future = to_device(contexts, device), to_device(future, device)
    quantiles = decode_with_grad(model, contexts, n_steps)
    return pinball_loss(quantiles, future, contexts.std(dim=1).clamp_min(1e-4))


def train_run(
    model, series, train_rows, n_steps, skip, horizon, device, *, loss_name, scope, steps, lr, batch_size, rng, context: int = CONTEXT
) -> dict:
    """A short fine-tuning run, with the pinball loss or a same-date correlation loss."""
    set_trainable(model, scope)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    # eligible_rows returns rows sorted by date.
    dates, starts = np.unique(train_rows[:, 0], return_index=True)
    by_date = dict(zip(dates, np.split(train_rows, starts[1:]))) if loss_name == "correlation" else {}
    early, late, started = [], [], time.time()
    skipped = 0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if loss_name == "pinball":
            sample = train_rows[rng.integers(0, len(train_rows), batch_size)]
            loss = batch_loss(model, series, sample, n_steps, device, context)
        else:
            pool = by_date[rng.choice(dates)]
            sample = pool[rng.choice(len(pool), min(batch_size, len(pool)), replace=False)]
            contexts, future = make_batch(series, sample, context, n_steps)
            quantiles = decode_with_grad(model, to_device(contexts, device), n_steps)
            score = five_day_score(quantiles, skip, horizon)
            target = to_device(future, device)[:, skip : skip + horizon].sum(dim=1)
            loss = correlation_loss(score, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        value = float(loss.detach())
        if not math.isfinite(value):
            return {"failed": f"non-finite loss at step {step}"}
        if not math.isfinite(float(grad_norm)):
            # Skipped before the optimizer step, so that it cannot corrupt the weights.
            skipped += 1
            if skipped > steps // 5:
                return {"failed": f"more than 20% of the steps had an invalid gradient, at step {step}"}
            continue
        optimizer.step()
        (early if step < steps // 4 else late if step >= 3 * steps // 4 else []).append(value)
    return {
        "first_quarter_loss": float(np.mean(early)),
        "last_quarter_loss": float(np.mean(late)),
        "skipped_steps": skipped,
        "seconds": round(time.time() - started, 1),
    }


def score_rows(model, series, rows, *, skip, horizon, device, context: int = CONTEXT, batch_size: int = 256) -> np.ndarray:
    """Five-day scores of (date, stock) rows. Only contexts are read, so no later return is touched."""
    scores = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            contexts, _ = make_batch(series, rows[start : start + batch_size], context, 0)
            quantiles = decode_with_grad(model, to_device(contexts, device), skip + horizon)
            scores.append(five_day_score(quantiles, skip, horizon).cpu().numpy())
    return np.concatenate(scores)
