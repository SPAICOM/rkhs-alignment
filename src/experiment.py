"""Configuration-driven plumbing shared by the experiment scripts.

Every entry point under ``scripts/`` needs the same few things before any
alignment happens: work out which encoder pairs to run, load exactly those
agents with their rows paired, give every receiver the private decoder
that measures post-alignment accuracy, and build the aligner named by the
config. That logic lives here so the scripts cannot drift apart on it.

Everything takes the Hydra ``DictConfig`` directly -- these are experiment
helpers, not library code, and the config *is* the experiment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig, OmegaConf

from .decoder import Decoder, MLPDecoder, TimmDecoder
from .latent import LatentSpace, load_agents

if TYPE_CHECKING:
    from .alignment import Aligner

log = logging.getLogger(__name__)

# Keys an alignment preset may carry that describe how the method should
# be *run* rather than how it should be *built*, and which therefore must
# be stripped before the constructor sees them.
ALIGNER_META_KEYS: tuple[str, ...] = ('pilot_strategy', 'rate_key')

__all__ = [
    'ALIGNER_META_KEYS',
    'apply_symbol_budget',
    'available_agents',
    'build_aligner',
    'build_decoder',
    'decoder_checkpoint',
    'load_pair_data',
    'method_pilot_strategy',
    'resolve_pairs',
]


# ---------------------------------------------------------------------
# Alignment presets
# ---------------------------------------------------------------------


def build_aligner(method_cfg: DictConfig, **overrides) -> Aligner:
    """Instantiate an alignment preset, minus its run-time metadata.

    Every preset under ``config/hydra/alignment/`` declares the pilot
    design the method expects (:func:`method_pilot_strategy`) alongside
    its hyper-parameters, because the two belong together: a method's
    published results assume a particular calibration design, and pairing
    them in one file is what stops a run silently evaluating a method
    under someone else's. Those keys are not constructor arguments,
    though, so they are dropped here rather than at each call site.
    """
    fields = {
        key: value
        for key, value in method_cfg.items()
        if key not in ALIGNER_META_KEYS
    }
    return hydra.utils.instantiate(fields, **overrides)


def apply_symbol_budget(cfg: DictConfig) -> dict[str, int]:
    """Point every method at the same channel rate.

    Methods buy their rate with different knobs -- an anchor-based
    equalizer transmits one coefficient per anchor, a coordinate method
    transmits its truncated whitened latent -- so matching them takes
    more than setting a single field. Each preset names its own knob in
    ``rate_key``; this writes ``cfg.symbols`` into whichever one that is,
    in place, before anything is instantiated.

    Without it a comparison at "the same k" is not a comparison at the
    same rate, and the cheaper method is being asked to do the same job
    on fewer symbols.

    Returns
    -------
    dict[str, int]
        The knob each method had set, for logging.
    """
    budget = cfg.get('symbols')
    if budget is None:
        return {}

    applied: dict[str, int] = {}
    for name, method_cfg in cfg.methods.items():
        key = method_cfg.get('rate_key')
        if key is None:
            log.warning(
                'Method %s declares no `rate_key`, so `symbols=%s` cannot '
                'be applied to it; its rate is whatever its preset says.',
                name,
                budget,
            )
            continue
        method_cfg[key] = int(budget)
        applied[name] = int(budget)
    return applied


def method_pilot_strategy(
    method_cfg: DictConfig, default: str | None = None
) -> str | None:
    """The pilot design a preset asks for, or ``default`` if it names none."""
    return str(method_cfg.get('pilot_strategy') or default or '') or None


def resolve_pairs(
    cfg: DictConfig, agent_names: list[str]
) -> list[tuple[str, str]]:
    """Turn the ``pairs`` / ``receiver`` config into explicit pairs.

    Parameters
    ----------
    cfg : DictConfig
        The run configuration.
    agent_names : list[str]
        Agents the data source offers, in config order.

    Returns
    -------
    list[tuple[str, str]]
        ``(source, target)`` model names.
    """
    if cfg.get('pairs'):
        pairs = [(str(p['source']), str(p['target'])) for p in cfg.pairs]
    else:
        # Star topology: everybody transmits to a single receiver.
        receiver = cfg.get('receiver') or agent_names[0]
        senders = [name for name in agent_names if name != receiver]
        max_pairs = cfg.get('max_pairs')
        if max_pairs:
            senders = senders[: int(max_pairs)]
        pairs = [(sender, receiver) for sender in senders]

    if not pairs:
        raise ValueError(
            'No alignment pairs to run: set `pairs`, or `receiver` plus a '
            'data source with at least two agents.'
        )
    for source, target in pairs:
        if source == target:
            raise ValueError(
                f'Pair ({source} -> {target}) aligns an encoder with '
                'itself; drop it from `pairs`.'
            )
    return pairs


def load_pair_data(
    cfg: DictConfig, pairs: list[tuple[str, str]]
) -> dict[str, dict[str, LatentSpace]]:
    """Load exactly the agents the resolved pairs reference."""
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    source = data_cfg.pop('source')
    data_cfg.pop('seed', None)
    data_cfg.pop('models', None)

    needed = sorted({name for pair in pairs for name in pair})
    return load_agents(source, models=needed, seed=cfg.seed, **data_cfg)


def available_agents(cfg: DictConfig) -> list[str]:
    """Agent names the data source offers, without loading any latents."""
    models = cfg.data.get('models')
    if models:
        return [str(m) for m in models]
    if cfg.data.source == 'synthetic':
        n_agents = int(cfg.data.get('n_agents', 8))
        width = max(2, len(str(n_agents - 1)))
        return [f'agent_{i:0{width}d}' for i in range(n_agents)]
    raise ValueError(
        f'Cannot enumerate agents for source {cfg.data.source!r}: list them '
        'under `data.models` or give explicit `pairs`.'
    )


# ---------------------------------------------------------------------
# Receiver decoders
# ---------------------------------------------------------------------


def decoder_checkpoint(cfg: DictConfig, model: str) -> Path:
    """Path of a cached decoder checkpoint for one receiver."""
    slug = model.replace('.', '_').replace('/', '_')
    dataset = cfg.data.get('dataset', cfg.data.source)
    return (
        Path(cfg.decoder.checkpoint_dir)
        / dataset
        / slug
        / f'seed{cfg.seed}.pt'
    )


def build_decoder(cfg: DictConfig, space: LatentSpace) -> Decoder:
    """Fit (or restore) the receiver's private decoder on its own latents.

    Parameters
    ----------
    cfg : DictConfig
        The run configuration.
    space : LatentSpace
        The receiver's *train* split; the decoder never sees any
        transported latent.

    Returns
    -------
    Decoder
    """
    if space.labels is None:
        raise ValueError(
            f'Agent {space.model_name!r} has no labels, so no decoder can be '
            'fitted. Set decoder.enabled=false to skip the downstream metric.'
        )

    if cfg.decoder.kind == 'linear':
        decoder = TimmDecoder(
            model_name=space.model_name,
            input_dim=space.dim,
            n_classes=int(cfg.decoder.n_classes),
            l2=float(cfg.decoder.l2),
        )
        return decoder.fit(space.latent, space.labels)

    if cfg.decoder.kind != 'mlp':
        raise ValueError(f'Unknown decoder kind {cfg.decoder.kind!r}.')

    path = decoder_checkpoint(cfg, space.model_name)
    if cfg.decoder.cache and path.exists():
        decoder = MLPDecoder.load(path)
        if decoder.input_dim == space.dim:
            log.info(
                'Restored decoder for %s from %s.', space.model_name, path
            )
            return decoder
        log.warning(
            'Checkpoint %s expects dim=%s but %s has dim=%d; refitting.',
            path,
            decoder.input_dim,
            space.model_name,
            space.dim,
        )

    log.info('Fitting an MLP decoder for %s.', space.model_name)
    decoder = MLPDecoder(
        model_name=space.model_name,
        input_dim=space.dim,
        seed=cfg.seed,
        **OmegaConf.to_container(cfg.decoder.mlp, resolve=True),
    )
    decoder.fit(space.latent, space.labels)
    if cfg.decoder.cache:
        decoder.save(path)
        log.info('Saved decoder checkpoint to %s.', path)
    return decoder
