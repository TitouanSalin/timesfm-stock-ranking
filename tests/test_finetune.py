"""Losses and window sampling for fine-tuning. Skipped where PyTorch is not installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qef.finetune import (  # noqa: E402
    QUANTILES,
    check_weights,
    correlation_loss,
    eligible_rows,
    five_day_score,
    make_batch,
    pinball_loss,
    score_rows,
    train_run,
)


class FakeTimesFM(torch.nn.Module):
    """A linear stand-in with the attributes and decode signature that fine-tuning uses."""

    def __init__(self, context):
        super().__init__()
        self.output_head = torch.nn.Linear(context, 64 * 9)
        self.transformer_stack = torch.nn.Module()
        self.transformer_stack.layers = torch.nn.ModuleList()

    def decode(self, target, horizon):
        return self.output_head(target[:, 0, :]).view(-1, 1, horizon, 9)


def noise(n_dates=120, n_stocks=30):
    return np.random.default_rng(0).normal(0, 0.02, (n_dates, n_stocks)).astype(np.float32)


def test_invalid_weights_stop_the_run():
    model = FakeTimesFM(16)
    check_weights(model)
    with torch.no_grad():
        model.output_head.weight[0, 0] = float("nan")
    with pytest.raises(SystemExit, match="1 weight tensors"):
        check_weights(model)


def test_training_reads_no_return_after_its_last_label():
    series = noise()
    history = np.tile(np.arange(1, 121)[:, None], (1, 30))
    rows = eligible_rows(np.ones(series.shape, dtype=bool), history, context=16, first=0, last=80)
    assert rows[:, 0].max() == 79
    # The last label is t+6 = 85. Reading anything later would make the loss invalid.
    series[86:] = np.nan
    outcome = train_run(
        FakeTimesFM(16), series, rows, 6, 1, 5, "cpu",
        loss_name="correlation", scope="head", steps=40, lr=1e-3, batch_size=8, rng=np.random.default_rng(0), context=16,
    )
    assert "failed" not in outcome


def test_training_reports_invalid_data_instead_of_going_on():
    series = noise()
    series[:] = np.nan
    history = np.tile(np.arange(1, 121)[:, None], (1, 30))
    rows = eligible_rows(np.ones(series.shape, dtype=bool), history, context=16, first=0, last=80)
    outcome = train_run(
        FakeTimesFM(16), series, rows, 6, 1, 5, "cpu",
        loss_name="correlation", scope="head", steps=4, lr=1e-3, batch_size=8, rng=np.random.default_rng(0), context=16,
    )
    assert "failed" in outcome


def test_scores_ignore_every_return_after_their_date():
    series = noise(60, 3)
    model = FakeTimesFM(16)
    rows = np.array([[50, 0], [50, 1], [59, 2]])  # The last date of the series has no future at all.
    scores = score_rows(model, series, rows, skip=1, horizon=5, device="cpu", context=16, batch_size=2)
    changed = series.copy()
    changed[51:, :2] = 99.0
    after = score_rows(model, changed, rows[:2], skip=1, horizon=5, device="cpu", context=16)
    assert scores[:2] == pytest.approx(after)


def test_pinball_loss_on_a_known_value():
    quantiles = torch.zeros(1, 1, 9)
    # With an error of +1 above every quantile, the loss is the mean of the levels.
    assert float(pinball_loss(quantiles, torch.ones(1, 1))) == pytest.approx(np.mean(QUANTILES))
    # Below every quantile, each level pays (1 - level).
    assert float(pinball_loss(quantiles, -torch.ones(1, 1))) == pytest.approx(np.mean([1 - q for q in QUANTILES]))


def test_pinball_loss_is_divided_by_each_window_scale():
    quantiles = torch.zeros(2, 1, 9)
    target = torch.tensor([[1.0], [1.0]])
    scaled = pinball_loss(quantiles, target, torch.tensor([2.0, 4.0]))
    assert float(scaled) == pytest.approx(0.5 * (0.5 / 2 + 0.5 / 4))


def test_correlation_loss_bounds():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert float(correlation_loss(x, 2 * x + 1)) == pytest.approx(-1.0, abs=1e-5)
    assert float(correlation_loss(x, -x)) == pytest.approx(1.0, abs=1e-5)


def test_correlation_loss_ignores_the_market_level():
    x = torch.tensor([1.0, 2.0, 4.0, 3.0])
    y = torch.tensor([0.1, 0.3, 0.2, 0.5])
    assert float(correlation_loss(x, y)) == pytest.approx(float(correlation_loss(x + 5, y + 7)), abs=1e-5)


def test_five_day_score_excludes_the_day_before_execution():
    quantiles = torch.zeros(1, 6, 9)
    quantiles[0, :, 4] = torch.tensor([100.0, 1, 1, 1, 1, 1])
    assert float(five_day_score(quantiles, skip=1, horizon=5)) == pytest.approx(5.0)


def test_batch_context_stops_at_the_date_and_labels_start_after_it():
    series = np.arange(60, dtype=float).reshape(20, 3)  # Value = 3 * date + stock.
    contexts, future = make_batch(series, np.array([[10, 2]]), context=4, n_steps=3)
    assert contexts[0].tolist() == [3 * t + 2 for t in (7, 8, 9, 10)]
    assert future[0].tolist() == [3 * t + 2 for t in (11, 12, 13)]


def test_eligible_rows_respect_the_range_mask_and_history():
    mask = np.zeros((10, 2), dtype=bool)
    mask[[2, 5, 8], 0] = True
    mask[5, 1] = True
    history = np.tile(np.arange(10)[:, None], (1, 2))
    history[:, 1] = 0  # The second stock never has a full context.
    rows = eligible_rows(mask, history, context=4, first=3, last=8)
    assert rows.tolist() == [[5, 0]]  # Date 2 is before the range and date 8 is after it.
