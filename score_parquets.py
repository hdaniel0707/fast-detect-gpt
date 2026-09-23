r"""Score the text column of one or more parquet files with Fast-DetectGPT.

Same contract as ``external/Binoculars/score_parquets.py``, deliberately: reads
each Parquet, works out per row which ones already hold a usable score in the
column the chosen ``--pair`` owns, reports the findings, asks whether existing
rows should be recomputed, then fills whatever is left. Files are written in
place unless ``-o`` is given (single input only), saved after every file so an
interrupt does not throw away finished work, and a file that fails mid-run is
reported and skipped while the rest continue.

**Exit status**, because a pipeline has nothing else to go on:

    0   every file that had work to do was scored and written, or there was
        nothing to do
    1   at least one file failed, or the confirmation prompt was declined

A skipped file is not success. The per-file ``except`` exists so one bad file
cannot lose a multi-hour run, not to make the run look clean.

--------------------------------------------------------------------------------
WHAT IT WRITES: THREE COLUMNS PER PAIR
--------------------------------------------------------------------------------
``fastdetect_crit_<pair>``   the conditional probability curvature. THE number.
                             Unitless, pair-specific, and **high for machine
                             text** -- the opposite direction to
                             ``binoculars_score``, which reads low. Anything
                             consuming both has to be told the direction.

``fastdetect_prob_<pair>``   P(machine) from upstream's Normal fit, and Null for
                             every pair upstream never fitted. Convenience only.
                             It is calibrated on the *paper's* dev set and
                             generators, not on this corpus, and it is
                             non-monotone in the far left tail (see below). Fit
                             a threshold on the criterion for anything that
                             matters.

``fastdetect_ntok_<pair>``   tokens actually scored, after truncation at
                             ``--max-tokens``. Cheap to write and the only way
                             to tell "the detector disagrees" apart from "the
                             detector saw 512 tokens of a 4000-token paper",
                             which is a live distinction on the science corpus.
                             Per pair, because the pairs do not share a
                             tokenizer.

--------------------------------------------------------------------------------
THE LEFT-TAIL TRAP IN THE PROBABILITY COLUMN
--------------------------------------------------------------------------------
Upstream maps criterion to probability by assuming a Normal per class and
setting ``sigma1 = 2 * sigma0``. A ratio of two Normals with unequal sd is a
sigmoid of a *quadratic*, not monotone: below a turning point roughly one sd
under the human mean, the wide machine-Normal takes over again and text that
looks *emphatically* human is assigned P(machine) -> 1. Upstream's own comment
says as much ("the probability could be high on both left side and right side").

For the falcon-7b constants the turning point is crit = -1.07. On a 6000-row
corpus there are always rows past it. This script counts them per file, prints
the count, and puts it in the summary JSON as ``prob_nonmonotone``. It does not
clamp them and does not drop them: the *criterion* for those rows is fine, and
only the probability mapping fails there. If that count is not ~0, do not use
the probability column for that file.

--------------------------------------------------------------------------------
BATCH SIZE IS 1, AND THAT IS NOT A TUNING KNOB
--------------------------------------------------------------------------------
``get_sampling_discrepancy_analytic`` opens with ``assert logits_ref.shape[0] ==
1``. The statistic standardises one document's log-likelihood against the mean
and variance of its *own* reference distribution, summed over that document's
tokens; batching would have to keep those reductions per row and mask the
padding out of all three of them. Upstream does not, so neither does this. The
cost is bounded anyway -- one forward pass per model per document, against
Binoculars' two, with no perturbation sampling at all. That is the whole point
of the method.

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------
Always from the parent repository's root, never from inside the submodule.
``--project`` points uv at THIS submodule's venv, which is separate from the
parent's on purpose (incompatible transformers pins):

    uv run --project external/fast-detect-gpt \
        python external/fast-detect-gpt/score_parquets.py \
        data/parquet/ghostbuster_gpt56luna.parquet \
        --pair falcon-7b --gpu 0 --cpu-threads 16 --yes

Prove the plumbing first on the single-model pair, which is ~5GiB of weights and
minutes per corpus rather than hours:

    uv run --project external/fast-detect-gpt \
        python external/fast-detect-gpt/score_parquets.py \
        data/parquet/ghostbuster_gpt56luna.parquet \
        --pair gptneo-2.7b --limit 50 -o /tmp/smoke.parquet --yes

Normally driven by the parent repository's pair runner, which loops the
pairs sequentially per file and checks the postcondition afterwards.

Fast-DetectGPT: Bao et al., ICLR 2024. Fork of github.com/baoguangsheng/fast-detect-gpt (MIT).
"""

from __future__ import annotations

# Standard library only until --gpu / --cpu-threads have been applied: both are
# read when the CUDA driver and the OpenMP runtime initialise, which happens
# inside torch, so setting them after `import torch` is too late.
import argparse
import os
from pathlib import Path

# Torch-free on purpose, for the same reason.
import detector_pairs
from detector_pairs import COLUMN_PREFIX, DEFAULT_MAX_TOKENS, DEFAULT_PAIR, PAIRS


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+",
                    help="Parquet file(s) and/or directories containing parquet files")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output path (default: overwrite each input in place). Only "
                         "accepted when the inputs resolve to a single file.")
    ap.add_argument("--text-column", default="text",
                    help="Name of the column containing text to score")
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug option: only score the first N rows of each file. Later "
                         "rows keep whatever they already have, they are not dropped.")
    ap.add_argument("--pair", default=DEFAULT_PAIR, choices=sorted(PAIRS),
                    help="Scoring pair from the registry in detector_pairs.py. Sets the "
                         "sampling model, the scoring model AND the columns together, so "
                         "two pairs never overwrite each other. Default: %(default)s, "
                         "which is the same two models as the existing binoculars_score "
                         "column and therefore the controlled comparison.")
    ap.add_argument("--sampling-model", default=None,
                    help="Sampling (reference) model, overriding --pair. Requires "
                         "--crit-column, since a pair off the registry owns no column.")
    ap.add_argument("--scoring-model", default=None,
                    help="Scoring (likelihood) model, overriding --pair. Requires "
                         "--crit-column.")
    ap.add_argument("--crit-column", default=None,
                    help="Column the criterion is written to. Default: the column the "
                         "chosen --pair owns. Criteria from different pairs are on "
                         "different scales and never share a column.")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="Truncation cap in tokens (default: %(default)s, matching the "
                         "Binoculars fork so the falcon-7b comparison holds the amount "
                         "of text fixed as well as the models).")
    ap.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"),
                    help="Model dtype (default: %(default)s, which is what upstream used "
                         "for these models and therefore what its calibration constants "
                         "were measured under). bfloat16 is the safer numeric choice on "
                         "an A100 but moves the probability column slightly off its "
                         "calibration; the criterion is unaffected either way.")
    ap.add_argument("--fp32-metrics", action=argparse.BooleanOptionalAction, default=True,
                    help="Compute the criterion in fp32 while the models still run in "
                         "--dtype (default: on). The statistic sums a log_softmax and "
                         "its square over the whole vocabulary; at 128k-152k terms in "
                         "fp16 that sum carries real noise. Costs one transient logits "
                         "copy per model.")
    ap.add_argument("--gpu", default=None,
                    help="value for CUDA_VISIBLE_DEVICES (e.g. '0'). Default: leave unset")
    ap.add_argument("--cpu-threads", type=int, default=None,
                    help="limit CPU threads (sets OMP_NUM_THREADS / MKL_NUM_THREADS)")
    ap.add_argument("--recompute", action="store_true",
                    help="Rescore rows that already have a criterion, without asking.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Decision threshold for counting verdict flips on a rescore "
                         "(crit >= threshold => machine-generated; note the direction is "
                         "the opposite of Binoculars). Default: the midpoint between the "
                         "pair's two calibrated means, or no flip counting at all when "
                         "the pair is uncalibrated.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the confirmation prompts (implies keeping existing scores "
                         "unless --recompute is also given).")
    ap.add_argument("--no-summary-file", action="store_true",
                    help="Print the summary but do not write the *_fastdetect_summary.json "
                         "files next to the outputs.")
    return ap.parse_args()


args = parse_args()


def resolve_pair(args: argparse.Namespace):
    """(sampling, scoring, crit col, prob col, ntok col, registry entry)."""
    pair = detector_pairs.resolve(args.pair)
    explicit = args.sampling_model is not None or args.scoring_model is not None
    if not explicit:
        return (pair.sampling, pair.scoring, args.crit_column or pair.crit_column,
                pair.prob_column, pair.crit_column.replace("_crit_", "_ntok_"), pair)

    sampling = args.sampling_model or pair.sampling
    scoring = args.scoring_model or pair.scoring
    known = detector_pairs.pair_for_models(sampling, scoring)
    if known is not None:
        return (sampling, scoring, args.crit_column or known.crit_column,
                known.prob_column, known.crit_column.replace("_crit_", "_ntok_"), known)
    if args.crit_column is None:
        raise SystemExit(
            f"--sampling-model/--scoring-model name a pair that is not in the registry "
            f"({sampling} + {scoring}), so there is no column it owns. Pass "
            f"--crit-column NAME (starting with {COLUMN_PREFIX!r}), or add the pair to "
            f"detector_pairs.py -- which is the better move if you intend to keep the "
            f"numbers."
        )
    # An off-registry pair has no calibration, so it gets no probability column:
    # a name derived from the criterion column would be a column of Nulls whose
    # only effect is to look like it should have values.
    #
    # The token-count name is derived by substitution, which is safe for every
    # registry column (they all contain "_crit_") and NOT safe for an arbitrary
    # --crit-column: with no "_crit_" in it the replace is a no-op, the two names
    # collide, and the run overwrites the criterion it just computed with token
    # counts. Suffix instead when the substitution does not bite.
    base = args.crit_column
    ntok = base.replace("_crit_", "_ntok_")
    if ntok == base:
        ntok = f"{base}_ntok"
    return sampling, scoring, base, None, ntok, None


SAMPLING, SCORING, CRIT_COL, PROB_COL, NTOK_COL, PAIR = resolve_pair(args)
SINGLE_MODEL = SAMPLING == SCORING

if not CRIT_COL.startswith(COLUMN_PREFIX):
    print(f"⚠️  Criterion column {CRIT_COL!r} does not start with {COLUMN_PREFIX!r}; "
          "the human-vs-AI analysis will not pick it up on its own.")

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.cpu_threads is not None:
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)

import json  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# The gated pairs (llama3-8b, llama31-8b) need HF_TOKEN, and it lives in the
# PARENT repo's .env -- this submodule has only a .env_sample. Loaded explicitly
# by path because the working directory for a run is the parent repo root, but
# nothing guarantees that and a silently missing token turns into a 401 half an
# hour into a download.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# How far a recomputed criterion may move and still count as the same number.
# The criterion is a standardised distance living roughly in [-3, +6], so these
# are absolute tolerances on that scale.
IDENTICAL_ABS = 1e-6
DRIFT_ABS = 1e-3

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


# ---------------------------------------------------------------------------
# The criterion, vendored from upstream.
# ---------------------------------------------------------------------------
# Copyright (c) Guangsheng Bao. MIT licence, see LICENSE in this directory.
# Verbatim from scripts/fast_detect_gpt.py::get_sampling_discrepancy_analytic.
#
# Copied rather than imported because that module imports data_builder and
# metrics at module level, which pull in datasets, sklearn and matplotlib --
# three heavy dependencies for twenty lines of pure torch, none of which this
# venv would otherwise need. The function is the paper's equation and does not
# move; if you pull upstream and it has changed, this copy is what to update.
def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()


class TokenizerMismatch(RuntimeError):
    """The two models do not tokenize the same text identically."""


class FastDetect:
    """The two models, and one document's criterion.

    Mirrors upstream's ``local_infer.FastDetectGPT`` -- same tokenization, same
    truncation, same criterion -- with three differences that matter at corpus
    scale rather than at a prompt: an explicit ``max_length`` instead of relying
    on whatever the tokenizer's ``model_max_length`` happens to be, an optional
    fp32 cast for the reduction, and a tokenizer check that fails up front on
    probe strings rather than on document 4000 of 6000.
    """

    def __init__(self, sampling: str, scoring: str, max_tokens: int,
                 dtype: torch.dtype, fp32_metrics: bool) -> None:
        self.max_tokens = max_tokens
        self.fp32_metrics = fp32_metrics
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading scoring model {scoring} ...", flush=True)
        self.scoring_tokenizer = self._load_tokenizer(scoring)
        self.scoring_model = self._load_model(scoring, dtype)

        if sampling == scoring:
            # One model doing both jobs: the reference distribution is the
            # scoring model's own. Upstream supports this and calibrates one
            # such pair; it halves the weights and the forward passes.
            self.sampling_tokenizer = self.scoring_tokenizer
            self.sampling_model = None
        else:
            print(f"Loading sampling model {sampling} ...", flush=True)
            self.sampling_tokenizer = self._load_tokenizer(sampling)
            self.sampling_model = self._load_model(sampling, dtype)
            self._check_tokenizers()

    def _load_tokenizer(self, name: str):
        tok = AutoTokenizer.from_pretrained(name, padding_side="right")
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        return tok

    def _load_model(self, name: str, dtype: torch.dtype):
        # transformers 5 renamed torch_dtype -> dtype. This fork pins >=5.12, so
        # the new name is the normal path; the fallback is there so the script
        # still runs if it is ever pointed at an older environment (upstream's
        # own requirements.txt pins 4.28.1).
        try:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
        model.to(self.device)
        model.eval()
        return model

    def _check_tokenizers(self) -> None:
        """Fail now, on probe strings, rather than on document 4000 of 6000.

        The criterion indexes the scoring model's log-probabilities with the
        token ids and compares them against the sampling model's distribution
        at the same positions. That is only meaningful if both models saw the
        *same* token sequence, which upstream enforces with a bare assert per
        document. A pair whose tokenizers differ produces either a crash
        thousands of rows in, or -- worse, when the lengths happen to agree --
        a column of numbers that mean nothing.
        """
        probes = [
            "The quick brown fox jumps over the lazy dog.",
            "In this paper we present a novel method for detecting machine-generated text.",
            'He said, "It\'s 42% — give or take."\n\nA second paragraph follows.',
        ]
        for probe in probes:
            a = self._encode(self.scoring_tokenizer, probe).input_ids
            b = self._encode(self.sampling_tokenizer, probe).input_ids
            if a.shape != b.shape or not torch.all(a == b):
                raise TokenizerMismatch(
                    f"{SAMPLING} and {SCORING} tokenize text differently "
                    f"({tuple(b.shape)} vs {tuple(a.shape)} ids on a probe string). "
                    f"Fast-DetectGPT feeds both models the same token sequence, so "
                    f"this pair cannot be scored. Pick a base/instruct pair of one "
                    f"family."
                )

    def _encode(self, tokenizer, text: str):
        return tokenizer(
            text,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
            padding=False,
            return_token_type_ids=False,
        )

    @torch.no_grad()
    def compute_crit(self, text: str) -> tuple[float, int]:
        """(criterion, tokens scored) for one document."""
        tokenized = self._encode(self.scoring_tokenizer, text).to(self.device)
        labels = tokenized.input_ids[:, 1:]
        if labels.size(1) < 1:
            # A single-token document has no next-token prediction to score.
            raise ValueError("text is too short to score (fewer than 2 tokens)")

        logits_score = self.scoring_model(**tokenized).logits[:, :-1]
        if self.sampling_model is None:
            logits_ref = logits_score
        else:
            ref_tokenized = self._encode(self.sampling_tokenizer, text).to(self.device)
            if not torch.all(ref_tokenized.input_ids[:, 1:] == labels):
                # Backstop for the up-front probe: a tokenizer can agree on
                # ASCII and diverge on a document full of typography.
                raise TokenizerMismatch("tokenizers disagree on this document")
            logits_ref = self.sampling_model(**ref_tokenized).logits[:, :-1]

        if self.fp32_metrics:
            # Order matters: in the single-model case logits_ref IS logits_score,
            # so cast once and re-alias. Casting logits_score first and then
            # testing identity would always miss, quietly allocating a second
            # full-size fp32 logits tensor for no numerical gain.
            shared = logits_ref is logits_score
            logits_score = logits_score.float()
            logits_ref = logits_score if shared else logits_ref.float()

        crit = get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)
        return crit, int(labels.size(1))


# ---------------------------------------------------------------------------
# Parquet inspection. Same shape as the Binoculars fork's, keyed on the
# criterion column -- prob and ntok are derived, so a row is "scored" iff it
# holds a criterion.
# ---------------------------------------------------------------------------
@dataclass
class FileStatus:
    path: Path
    n_rows: int
    n_window: int
    exists: bool
    n_filled: int
    n_missing: int
    n_blank: int
    n_filled_total: int
    n_blank_scored: int
    scoreable: np.ndarray

    @property
    def state(self) -> str:
        if not self.exists:
            return "absent"
        if self.n_filled == 0:
            return "empty"
        if self.n_missing > 0:
            return "partial"
        return "complete"


def _is_empty_text(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _as_score(value) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(score) else score


def collect_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.parquet")))
        else:
            paths.append(p)
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return unique


def check_parquet(df: pd.DataFrame, text_col: str, path: Path) -> np.ndarray:
    if len(df) == 0:
        raise ValueError(f"{path} has no rows.")
    if text_col not in df.columns:
        raise ValueError(
            f"Column {text_col!r} not found in {path}. Available columns: {list(df.columns)}"
        )
    return np.array([not _is_empty_text(v) for v in df[text_col]])


def window_mask(n_rows: int, limit: int | None) -> np.ndarray:
    window = np.ones(n_rows, dtype=bool)
    if limit is None:
        return window
    if limit < 1:
        raise ValueError(f"--limit must be >= 1, got {limit}")
    window[limit:] = False
    return window


def inspect(path: Path, text_col: str, limit: int | None) -> tuple[pd.DataFrame, FileStatus]:
    df = pd.read_parquet(path)
    nonblank = check_parquet(df, text_col, path)
    window = window_mask(len(df), limit)
    scoreable = nonblank & window

    if CRIT_COL not in df.columns:
        filled = np.zeros(len(df), dtype=bool)
        exists = False
    else:
        filled = np.array([_as_score(v) is not None for v in df[CRIT_COL]])
        exists = True

    return df, FileStatus(
        path=path,
        n_rows=len(df),
        n_window=int(window.sum()),
        exists=exists,
        n_filled=int((window & nonblank & filled).sum()),
        n_missing=int((window & nonblank & ~filled).sum()),
        n_blank=int((window & ~nonblank).sum()),
        n_filled_total=int(filled.sum()),
        n_blank_scored=int((~nonblank & filled).sum()),
        scoreable=scoreable,
    )


def positions_to_score(df: pd.DataFrame, st: FileStatus, recompute_existing: bool) -> np.ndarray:
    if recompute_existing or not st.exists:
        return np.flatnonzero(st.scoreable)
    filled = np.array([_as_score(v) is not None for v in df[CRIT_COL]])
    return np.flatnonzero(st.scoreable & ~filled)


def describe_device() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available():
        return "CPU (no CUDA available) — this will be extremely slow"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    suffix = f", CUDA_VISIBLE_DEVICES={visible}" if visible else ""
    return f"cuda:0 — {name}, {total:.0f} GiB{suffix}"


def default_threshold() -> float | None:
    """Midpoint between the calibrated class means, or None.

    Only used to count how many verdicts a *rescore* flipped, so an approximate
    cut is fine and an absent one simply means that count is not reported. It is
    emphatically not an operating point for this corpus.
    """
    if args.threshold is not None:
        return args.threshold
    if PAIR is None or PAIR.calib is None:
        return None
    return (PAIR.calib.mu0 + PAIR.calib.mu1) / 2


THRESHOLD = default_threshold()


def print_findings(statuses: list[FileStatus], text_col: str) -> None:
    print()
    print("=" * 94)
    print("Fast-DetectGPT")
    print(f"  files          : {len(statuses)}")
    print(f"  text column    : {text_col!r}")
    print(f"  criterion col  : {CRIT_COL!r}   (HIGH = machine, opposite to binoculars_score)")
    print(f"  probability col: {PROB_COL!r}" if PROB_COL else
          "  probability col: (none — unregistered pair, no calibration)")
    print(f"  tokens col     : {NTOK_COL!r}")
    print(f"  pair           : {PAIR.key if PAIR else 'unregistered'}")
    print(f"  sampling model : {SAMPLING}")
    print(f"  scoring model  : {SCORING}" + ("  (same model — single forward pass)" if SINGLE_MODEL else ""))
    print(f"  max tokens     : {args.max_tokens}")
    print(f"  dtype          : {args.dtype}"
          f"{' , criterion in fp32' if args.fp32_metrics else ''}")
    print(f"  device         : {describe_device()}")
    print(f"  cpu threads    : {args.cpu_threads if args.cpu_threads is not None else 'unset'}")
    if PAIR is not None and PAIR.calib is None:
        print(f"  {'':15}⚠️  uncalibrated pair: {PROB_COL!r} will be left Null. "
              f"Use the criterion.")
    if PAIR is not None and PAIR.gated:
        print(f"  {'':15}NOTE gated on the Hub — needs HF_TOKEN and an accepted licence.")
    print("=" * 94)
    print(f"{'file':<38}{'state':<10}{'rows':>7}{'window':>8}{'scored':>8}"
          f"{'missing':>9}{'blank':>7}  note")
    print("-" * 94)
    for st in statuses:
        note = ""
        if st.n_blank_scored:
            note = f"{st.n_blank_scored} blank row(s) carry a stale score"
        elif st.n_blank:
            note = "blank text stays Null"
        name = st.path.name
        if len(name) > 36:
            name = "..." + name[-33:]
        print(f"{name:<38}{st.state:<10}{st.n_rows:>7}{st.n_window:>8}{st.n_filled:>8}"
              f"{st.n_missing:>9}{st.n_blank:>7}  {note}")
    print("-" * 94)

    total_blank = sum(st.n_blank for st in statuses)
    if total_blank:
        print(f"❌ {total_blank} empty/blank row(s) across all files (score left Null)")
    else:
        print("✅ Every row has a non-empty text.")
    total_stale = sum(st.n_blank_scored for st in statuses)
    if total_stale:
        print(f"⚠️  {total_stale} row(s) with blank text already carry a score. They are "
              "left exactly as they are — this run neither rescores nor clears them.")


def ask_yes_no(question: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = input(question + suffix).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def plan(statuses: list[FileStatus], recompute_existing: bool) -> list[FileStatus]:
    if recompute_existing:
        return [st for st in statuses if st.n_filled > 0 or st.n_missing > 0]
    return [st for st in statuses if st.n_missing > 0]


def score_positions(det: FastDetect, df: pd.DataFrame, text_col: str,
                    positions: np.ndarray) -> tuple[pd.DataFrame, list, list, dict]:
    """Score the given rows and write all three columns.

    Returns (df, old_crits, new_crits, per-file counters). A row that raises --
    too short to score, or a tokenizer disagreement on that particular document
    -- keeps a Null criterion and is counted, rather than losing the file. That
    matches how a blank row is treated: no score is a legitimate outcome for a
    row, and the summary says how many.
    """
    n = len(df)
    crits: list = list(df[CRIT_COL]) if CRIT_COL in df.columns else [float("nan")] * n
    probs: list = list(df[PROB_COL]) if (PROB_COL and PROB_COL in df.columns) else [float("nan")] * n
    ntoks: list = list(df[NTOK_COL]) if NTOK_COL in df.columns else [None] * n

    old_crits = [_as_score(crits[pos]) for pos in positions]
    texts = df[text_col].astype(str).to_numpy()
    new_crits: list = []
    counters = {"unscorable": 0, "truncated": 0, "prob_nonmonotone": 0}
    prob_floor = PAIR.prob_min_crit if PAIR is not None else None

    for pos in tqdm(positions, desc=df.attrs.get("name", "scoring"),
                    unit="doc", leave=False):
        try:
            crit, ntok = det.compute_crit(texts[pos])
        except (ValueError, TokenizerMismatch) as exc:
            counters["unscorable"] += 1
            tqdm.write(f"  row {pos}: not scored ({exc})")
            new_crits.append(None)
            continue

        crits[pos] = crit
        ntoks[pos] = ntok
        new_crits.append(crit)
        if ntok >= det.max_tokens - 1:
            counters["truncated"] += 1
        if PAIR is not None and PROB_COL:
            prob = PAIR.prob_from_crit(crit)
            if prob is not None:
                probs[pos] = prob
                if prob_floor is not None and crit < prob_floor:
                    counters["prob_nonmonotone"] += 1

    df[CRIT_COL] = pd.Series(crits, index=df.index, dtype="float64")
    if PROB_COL:
        df[PROB_COL] = pd.Series(probs, index=df.index, dtype="float64")
    # Nullable integer: an unscored row has no token count, and a float column
    # of token counts would be a lie about what the number is.
    df[NTOK_COL] = pd.Series(ntoks, index=df.index, dtype="Int64")
    return df, old_crits, new_crits, counters


def delta_summary(old_values: list, new_values: list) -> dict[str, object]:
    """Classify each rescored row by how far the criterion moved."""
    stats: dict[str, object] = {
        "compared": 0, "new": 0, "lost": 0,
        "identical": 0, "drifted": 0, "changed": 0,
        "flip_to_machine": 0, "flip_to_human": 0,
    }
    deltas: list[float] = []
    for old, new in zip(old_values, new_values):
        if new is None:
            if old is not None:
                stats["lost"] += 1
            continue
        if old is None:
            stats["new"] += 1
            continue
        stats["compared"] += 1
        delta = new - old
        deltas.append(delta)
        if abs(delta) <= IDENTICAL_ABS:
            stats["identical"] += 1
        elif abs(delta) <= DRIFT_ABS:
            stats["drifted"] += 1
        else:
            stats["changed"] += 1
        if THRESHOLD is not None:
            was_machine = old >= THRESHOLD
            is_machine = new >= THRESHOLD
            if is_machine and not was_machine:
                stats["flip_to_machine"] += 1
            elif was_machine and not is_machine:
                stats["flip_to_human"] += 1
    if deltas:
        stats["abs_delta_max"] = float(np.max(np.abs(deltas)))
        stats["abs_delta_mean"] = float(np.mean(np.abs(deltas)))
    return stats


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "sd": float(arr.std()),
    }


def summary_payload(st: FileStatus, out_path: Path, n_scored: int, seconds: float,
                    deltas: dict, counters: dict, new_crits: list,
                    recompute_existing: bool) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detector": "fast-detect-gpt",
        "input": str(st.path),
        "output": str(out_path),
        "pair": PAIR.key if PAIR else "unregistered",
        "sampling_model": SAMPLING,
        "scoring_model": SCORING,
        "single_model": SINGLE_MODEL,
        "criterion_column": CRIT_COL,
        "probability_column": PROB_COL,
        "tokens_column": NTOK_COL,
        "calibrated": bool(PAIR is not None and PAIR.calib is not None),
        "direction": "high criterion = machine-generated",
        "max_tokens": args.max_tokens,
        "dtype": args.dtype,
        "fp32_metrics": args.fp32_metrics,
        "flip_threshold": THRESHOLD,
        "limit": args.limit,
        "recomputed_existing": recompute_existing,
        "rows_total": st.n_rows,
        "rows_in_window": st.n_window,
        "rows_blank": st.n_blank,
        "rows_scored": n_scored,
        "rows_unscorable": counters["unscorable"],
        "rows_truncated_at_cap": counters["truncated"],
        "prob_nonmonotone": counters["prob_nonmonotone"],
        "prob_floor_crit": PAIR.prob_min_crit if PAIR else None,
        "seconds": round(seconds, 1),
        "docs_per_second": round(n_scored / seconds, 2) if seconds > 0 else None,
        "criterion_distribution": _distribution(
            [v for v in new_crits if v is not None]
        ),
        "delta": deltas,
    }


def summary_path_for(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}__{CRIT_COL}_fastdetect_summary.json")


def save_summary(payload: dict, out_path: Path) -> Path:
    p = summary_path_for(out_path)
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return p


def print_file_stats(payload: dict, indent: str = "      ") -> None:
    dist = payload["criterion_distribution"]
    if dist:
        print(f"{indent}criterion   min {dist['min']:+.3f}  median {dist['median']:+.3f}  "
              f"max {dist['max']:+.3f}  (mean {dist['mean']:+.3f}, sd {dist['sd']:.3f})")
    if payload["rows_truncated_at_cap"]:
        pct = 100 * payload["rows_truncated_at_cap"] / max(payload["rows_scored"], 1)
        print(f"{indent}⚠️  {payload['rows_truncated_at_cap']} row(s) ({pct:.0f}%) hit the "
              f"{payload['max_tokens']}-token cap — those are scored on a prefix.")
    if payload["rows_unscorable"]:
        print(f"{indent}⚠️  {payload['rows_unscorable']} row(s) could not be scored "
              f"(left Null).")
    if payload["prob_nonmonotone"]:
        print(f"{indent}❌ {payload['prob_nonmonotone']} row(s) fall below crit "
              f"{payload['prob_floor_crit']:+.3f}, where the probability mapping turns "
              f"back upwards.\n{indent}   Their {PROB_COL!r} is high for text the "
              f"criterion calls emphatically human. Use the criterion column.")
    d = payload["delta"]
    if d["compared"]:
        print(f"{indent}rescored    identical {d['identical']}  drifted {d['drifted']}  "
              f"changed {d['changed']}")
        if THRESHOLD is not None and (d["flip_to_machine"] or d["flip_to_human"]):
            print(f"{indent}verdict     -> machine {d['flip_to_machine']}, "
                  f"-> human {d['flip_to_human']}  (at crit {THRESHOLD:+.3f})")


def print_summary(done: list, failed: list, summaries: list[Path]) -> None:
    print()
    print("=" * 94)
    print("Done")
    for st, out_path, payload in done:
        print(f"  ✅ {st.path.name}  {payload['rows_scored']} row(s) in "
              f"{payload['seconds']}s ({payload['docs_per_second']} doc/s) -> {out_path}")
        print_file_stats(payload)
    for path, reason in failed:
        print(f"  ❌ {path.name}  {reason}")
    if summaries:
        print()
        print("Summaries:")
        for p in summaries:
            print(f"  {p}")
    print("=" * 94)


def validate_out_path(paths: list[Path]) -> None:
    if args.out is not None and len(paths) > 1:
        raise SystemExit(
            f"-o/--out takes a single output path, but the inputs resolve to "
            f"{len(paths)} files. Run them one at a time, or drop -o to write in place."
        )


def main() -> None:
    paths = collect_paths(args.inputs)
    if not paths:
        raise SystemExit("No parquet files found to score.")
    validate_out_path(paths)

    frames: dict[Path, pd.DataFrame] = {}
    statuses: list[FileStatus] = []
    for path in paths:
        df, st = inspect(path, args.text_column, args.limit)
        df.attrs["name"] = path.name
        frames[path] = df
        statuses.append(st)

    print_findings(statuses, args.text_column)
    if args.limit is not None:
        print(f"NOTE  --limit {args.limit}: only the first {args.limit} row(s) of each file "
              "are considered and the counts above are within that window. Rows past the "
              "limit keep their current values and are still written to the output.")

    n_present = sum(1 for st in statuses if st.n_filled > 0)
    recompute_existing = args.recompute
    if n_present and not recompute_existing and not args.yes:
        recompute_existing = ask_yes_no(
            f"{n_present} file(s) already have scores. Rescore the rows that have one?",
            default=False,
        )

    todo = plan(statuses, recompute_existing)
    if not todo:
        print("\nEverything is already scored — nothing to do.")
        return

    print()
    print(f"Will score {len(todo)} file(s):")
    total_positions = 0
    for st in todo:
        positions = positions_to_score(frames[st.path], st, recompute_existing)
        total_positions += len(positions)
        print(f"  {st.path.name:<38} {len(positions):>7} rows  ({st.state})  "
              f"-> {args.out or st.path}")
    print(f"  {'total':<38} {total_positions:>7} rows")

    in_place = [st.path for st in todo if args.out is None]
    if in_place:
        print()
        print(f"NOTE  {len(in_place)} file(s) will be overwritten IN PLACE. "
              "Pass -o to write elsewhere (single file only).")

    if not args.yes:
        answer = input("Proceed? Type 'yes': ").strip().lower()
        if answer != "yes":
            raise SystemExit("Aborted — no changes made.")

    det = FastDetect(
        sampling=SAMPLING,
        scoring=SCORING,
        max_tokens=args.max_tokens,
        dtype=_DTYPES[args.dtype],
        fp32_metrics=args.fp32_metrics,
    )

    done: list = []
    failed: list = []
    written: list[Path] = []

    for st in tqdm(todo, desc="Files", unit="file"):
        positions = positions_to_score(frames[st.path], st, recompute_existing)
        if len(positions) == 0:
            continue

        out_path = args.out or st.path
        tqdm.write(f"→ {st.path.name}  ({len(positions)} rows)")
        started = time.perf_counter()
        try:
            df, old_crits, new_crits, counters = score_positions(
                det, frames[st.path], args.text_column, positions
            )
        except Exception as exc:  # one bad file must not lose the rest
            failed.append((st.path, f"{type(exc).__name__}: {exc}"))
            tqdm.write(f"  failed: {type(exc).__name__}: {exc}")
            continue

        elapsed = time.perf_counter() - started
        payload = summary_payload(
            st, out_path, len(positions), elapsed,
            delta_summary(old_crits, new_crits), counters, new_crits,
            recompute_existing,
        )

        # Save after every file so an interrupt does not lose finished work.
        df.to_parquet(out_path, index=False)
        done.append((st, out_path, payload))
        if not args.no_summary_file:
            written.append(save_summary(payload, out_path))

    print_summary(done, failed, written)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
