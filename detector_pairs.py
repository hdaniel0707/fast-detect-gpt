"""The model pairs Fast-DetectGPT can be run with, and what each one costs.

Fast-DetectGPT scores a document with a *sampling* model and a *scoring* model.
The sampling model supplies the reference distribution the observed tokens are
measured against; the scoring model supplies the likelihoods. The statistic is
the conditional probability curvature: how far the scoring model's likelihood of
the text sits above the mean it would expect under the sampling model's own
distribution, in units of that distribution's standard deviation.

This is NOT the Binoculars statistic, which is why the columns are worth having
side by side. Binoculars is a ratio of two perplexities and reads *low* for
machine text. Fast-DetectGPT is a standardised distance and reads *high* for
machine text. Same two models, opposite sign, different quantity:

    binoculars_score          machine text scores LOW   (threshold ~0.90)
    fastdetect_crit_*         machine text scores HIGH  (threshold ~ mid-gap)

Anything that consumes these columns has to know that. `analyse_score_human_vs_ai`
and the `score_baselines:` blocks in cfg/ai_detection/train/ both take a
threshold and a direction per column, so put the direction in when adding one.

This module is deliberately free of torch, so that score_parquets.py can resolve
a pair -- and reject a bad --pair -- before torch is imported, which is what lets
--gpu still take effect. Same arrangement as external/Binoculars/model_pairs.py.

CALIBRATION IS PER PAIR AND MOSTLY ABSENT
-----------------------------------------
The raw criterion is unitless and its scale depends on both models, so a
criterion from one pair means nothing next to a criterion from another. Upstream
turns it into a probability by assuming the criterion is Normal under each class
and fitting (mu0, sigma0) on human text and (mu1, sigma1) on machine text --
constants measured once, on the paper's own dev set, and hardcoded in
scripts/local_infer.py. They exist for four pairs and nothing else.

So `prob_column` is filled only where `calib` is not None, and stays Null
otherwise. That is the honest outcome: an uncalibrated pair has a perfectly good
criterion and no probability, and inventing one by borrowing another pair's
constants would produce a number that looks like a probability, sorts correctly,
and is wrong everywhere it matters -- which is precisely the failure the
uncalibrated Binoculars pairs already have to carry a footnote about.

Note also *whose* dev set those constants came from: the paper's, on the paper's
generators, in 2023-24. They are a reasonable prior and not a calibration for
this corpus. If you want a real operating point on this data, fit the threshold
on the criterion column and say so.
"""

# NOTE: deliberately no `from __future__ import annotations`.
#
# It would make every annotation a string, and @dataclass then resolves those
# through sys.modules[cls.__module__] -- which does not exist when this file is
# loaded by path with importlib.util.module_from_spec, exactly as
# run_fastdetect_pairs.load_registry does it. The failure is an AttributeError
# inside dataclasses, thrown at import, nowhere near the cause.
# `Calibration | None` below needs no future import on Python >= 3.10 anyway.
import math
from dataclasses import dataclass

# Truncation cap in tokens.
#
# 512 is chosen to match external/Binoculars/model_pairs.DEFAULT_MAX_TOKENS, not
# because Fast-DetectGPT wants it -- upstream's local_infer.py just lets the
# tokenizer truncate at the model's own maximum, which is 2048 for Falcon. The
# whole point of running the falcon-7b pair here is that it is the SAME two
# models as the existing binoculars_score column, so the only thing that differs
# between the two numbers is the statistic. Feeding one of them twice as much
# text would quietly reintroduce a second difference and make the comparison
# uninterpretable. Override per run with --max-tokens; if you raise it, raise it
# for both detectors or say which is which in the writeup.
DEFAULT_MAX_TOKENS = 512

_GIB = 1024 ** 3


@dataclass(frozen=True)
class Calibration:
    """Normal parameters mapping a criterion to P(machine), from upstream.

    Lifted verbatim from scripts/local_infer.py, where they sit in a dict with
    the accuracy each one reached on the paper's dev set. Copied rather than
    imported because that module pulls in scipy at import time for a single pdf
    call, and this one is meant to import cleanly with nothing installed.
    """

    mu0: float  # human-text mean
    sigma0: float  # human-text sd
    mu1: float  # machine-text mean
    sigma1: float  # machine-text sd
    acc: float  # accuracy upstream measured with these constants


@dataclass(frozen=True)
class Pair:
    """One sampling/scoring pair and the two columns its numbers belong in."""

    key: str  # --pair value
    sampling: str  # reference distribution ("sampling model" upstream)
    scoring: str  # likelihoods ("scoring model" upstream)
    crit_column: str  # raw conditional probability curvature
    prob_column: str  # P(machine) -- filled only when `calib` is set
    params: float  # parameters per model, in billions
    vocab: int
    year: str
    calib: Calibration | None = None
    gated: bool = False  # needs HF_TOKEN with an accepted licence
    note: str = ""

    @property
    def single_model(self) -> bool:
        """One model doing both jobs -- half the weights, one forward pass.

        Upstream supports this and calibrates one such pair. The reference
        distribution is then the scoring model's own, which is a weaker
        statistic but a genuinely cheap one.
        """
        return self.sampling == self.scoring

    @property
    def weights_gib(self) -> float:
        """Resident weights, fp16, counting the models actually loaded."""
        n_models = 1 if self.single_model else 2
        return n_models * self.params * 1e9 * 2 / _GIB

    def transient_gib(self, max_tokens: int = DEFAULT_MAX_TOKENS) -> float:
        """Peak logits working set for one document.

        Batch is always 1 -- the criterion asserts it (see score_parquets) -- so
        this is per document, not per batch. Counted as four logits-sized fp32
        copies summed over both models: the two logits tensors, the reference
        softmax, and the working copies inside the mean/variance reduction.
        """
        n_models = 1 if self.single_model else 2
        return (max_tokens * self.vocab * 4 * 4 * n_models) / _GIB

    def total_gib(self, max_tokens: int = DEFAULT_MAX_TOKENS) -> float:
        return self.weights_gib + self.transient_gib(max_tokens)

    def fits(self, device_gib: float, max_tokens: int = DEFAULT_MAX_TOKENS,
             slack_gib: float = 2.0) -> bool:
        return self.total_gib(max_tokens) + slack_gib <= device_gib

    @property
    def prob_min_crit(self) -> float | None:
        """Criterion below which P(machine) starts rising again, or None.

        Upstream fixes sigma1 = 2 * sigma0 "to make sure of a wider coverage of
        potential AI texts", and notes the consequence in a comment: "the
        probability could be high on both left side and right side of
        Normal(mu0, sigma0)". The ratio of two Normals with unequal sd is not
        monotone -- it is a sigmoid of a quadratic -- so below the turning point
        the wide machine-Normal wins again and *very human-looking* text is
        assigned P(machine) -> 1.

        For the falcon-7b constants that turning point is at crit ~ -1.07, which
        is only about one sd below the human mean. On an interactive demo you
        will never see it. On a 6000-row corpus there are always rows out there,
        and they arrive labelled with a confident, badly wrong probability.

        The turning point is where d/dx of the log ratio vanishes:

            x* = (mu1/sigma1^2 - mu0/sigma0^2) / (1/sigma1^2 - 1/sigma0^2)

        which is a minimum of the probability precisely because sigma1 > sigma0.
        score_parquets.py counts the rows below it and says so; it does not
        clamp or drop them, because the criterion for those rows is perfectly
        good and it is only the probability mapping that fails there. Use the
        criterion column, not the probability column, for anything quantitative.
        """
        if self.calib is None:
            return None
        c = self.calib
        inv0 = 1.0 / (c.sigma0 ** 2)
        inv1 = 1.0 / (c.sigma1 ** 2)
        if math.isclose(inv1, inv0):
            return None  # equal sd -> the log ratio is linear, prob is monotone
        return (c.mu1 * inv1 - c.mu0 * inv0) / (inv1 - inv0)

    def prob_from_crit(self, crit: float) -> float | None:
        """P(machine) for one criterion, or None when the pair is uncalibrated.

        The arithmetic upstream does with scipy.stats.norm, done in log space
        with the stdlib instead:

            p1/(p0+p1) = 1/(1 + p0/p1) = sigmoid(-log(p0/p1))

        which avoids both the scipy dependency and the underflow that the
        direct pdf ratio hits several sigma out -- where, on a corpus of
        thousands of documents, there are always some rows.
        """
        if self.calib is None:
            return None
        c = self.calib
        log_ratio = (
            math.log(c.sigma1 / c.sigma0)
            - ((crit - c.mu0) ** 2) / (2 * c.sigma0 ** 2)
            + ((crit - c.mu1) ** 2) / (2 * c.sigma1 ** 2)
        )
        # sigmoid(-log_ratio), written so neither branch can overflow exp().
        if log_ratio >= 0:
            return 1.0 / (1.0 + math.exp(log_ratio))
        return math.exp(-log_ratio) / (1.0 + math.exp(-log_ratio))


# Ordered by intended use: the controlled comparison first, then the pair the
# authors now recommend, then the paper's own defaults, then the uncalibrated
# pairs that mirror the Binoculars registry.
PAIRS: dict[str, Pair] = {
    "falcon-7b": Pair(
        key="falcon-7b",
        sampling="tiiuae/falcon-7b",
        scoring="tiiuae/falcon-7b-instruct",
        crit_column="fastdetect_crit_falcon_7b",
        prob_column="fastdetect_prob_falcon_7b",
        params=7.22, vocab=65024, year="2023",
        calib=Calibration(mu0=-0.0707, sigma0=0.9520, mu1=2.9306, sigma1=1.9039,
                          acc=0.8938),
        note="THE ONE TO RUN FIRST. Exactly the two models behind the existing "
             "binoculars_score column, so crit vs binoculars_score is a "
             "controlled comparison of the statistic with the models held "
             "fixed. Also upstream's local_infer default, and its best "
             "calibrated pair.",
    ),
    "llama3-8b": Pair(
        key="llama3-8b",
        sampling="meta-llama/Meta-Llama-3-8B",
        scoring="meta-llama/Meta-Llama-3-8B-Instruct",
        crit_column="fastdetect_crit_llama3_8b",
        prob_column="fastdetect_prob_llama3_8b",
        params=8.03, vocab=128256, year="2024",
        calib=Calibration(mu0=0.1603, sigma0=1.0791, mu1=2.4686, sigma1=2.1582,
                          acc=0.0),  # upstream ships no accuracy for this one
        gated=True,
        note="What the authors recommend as of 31/01/2026, and the only other "
             "calibrated pair worth the GPU time. NOTE Llama-3, not Llama-3.1: "
             "the calibration key is llama3-8b_llama3-8b-instruct, so the "
             "Llama-3.1 weights already cached for binoculars_score_llama31_8b "
             "do NOT satisfy it and this pair is a fresh ~32GB download. "
             "Gated: needs HF_TOKEN and an accepted licence on both repos.",
    ),
    "gptj-neo": Pair(
        key="gptj-neo",
        sampling="EleutherAI/gpt-j-6B",
        scoring="EleutherAI/gpt-neo-2.7B",
        crit_column="fastdetect_crit_gptj_neo",
        prob_column="fastdetect_prob_gptj_neo",
        params=6.05, vocab=50257, year="2021",
        calib=Calibration(mu0=0.2713, sigma0=0.9366, mu1=2.2334, sigma1=1.8731,
                          acc=0.8122),
        note="The paper's headline configuration (main.sh). Both models share "
             "the GPT-2 tokenizer, which the tokenizer check requires. Weakest "
             "of the calibrated pairs and the oldest -- it is here for "
             "faithfulness to the paper, not because it is a good 2026 proxy.",
    ),
    "gptneo-2.7b": Pair(
        key="gptneo-2.7b",
        sampling="EleutherAI/gpt-neo-2.7B",
        scoring="EleutherAI/gpt-neo-2.7B",
        crit_column="fastdetect_crit_gptneo_2_7b",
        prob_column="fastdetect_prob_gptneo_2_7b",
        params=2.65, vocab=50257, year="2021",
        calib=Calibration(mu0=-0.2489, sigma0=0.9968, mu1=1.8983, sigma1=1.9935,
                          acc=0.8222),
        note="SMOKE TEST. One model doing both jobs: ~5GiB of weights, one "
             "forward pass per document, minutes per corpus. Calibrated, so it "
             "exercises the whole path including the probability column. Prove "
             "the plumbing here before booking the A100 for a 7B pair.",
    ),
    # ---- uncalibrated: criterion only, prob_column stays Null -----------------
    # These mirror external/Binoculars/model_pairs.py so the same models can be
    # compared under both statistics. Upstream publishes no Normal parameters
    # for any of them, and this module will not borrow another pair's.
    "qwen25-7b": Pair(
        key="qwen25-7b",
        sampling="Qwen/Qwen2.5-7B",
        scoring="Qwen/Qwen2.5-7B-Instruct",
        crit_column="fastdetect_crit_qwen25_7b",
        prob_column="fastdetect_prob_qwen25_7b",
        params=7.62, vocab=152064, year="2024",
        note="Mirrors binoculars_score_qwen25_7b: science-heavy pretrain, the "
             "pair that was meant to fix the science corpus.",
    ),
    "falcon3-7b": Pair(
        key="falcon3-7b",
        sampling="tiiuae/Falcon3-7B-Base",
        scoring="tiiuae/Falcon3-7B-Instruct",
        crit_column="fastdetect_crit_falcon3_7b",
        prob_column="fastdetect_prob_falcon3_7b",
        params=7.46, vocab=131072, year="2024",
        note="Mirrors binoculars_score_falcon3_7b: the baseline family one "
             "generation on, so only the age of the pair changed.",
    ),
    "mistral-v03": Pair(
        key="mistral-v03",
        sampling="mistralai/Mistral-7B-v0.3",
        scoring="mistralai/Mistral-7B-Instruct-v0.3",
        crit_column="fastdetect_crit_mistral_v03",
        prob_column="fastdetect_prob_mistral_v03",
        params=7.25, vocab=32768, year="2024",
        note="Mirrors binoculars_score_mistral_v03: the general-web control, "
             "and by far the cheapest 7B pair at a 32k vocabulary.",
    ),
    "llama31-8b": Pair(
        key="llama31-8b",
        sampling="meta-llama/Llama-3.1-8B",
        scoring="meta-llama/Llama-3.1-8B-Instruct",
        crit_column="fastdetect_crit_llama31_8b",
        prob_column="fastdetect_prob_llama31_8b",
        params=8.03, vocab=128256, year="2024",
        gated=True,
        note="Mirrors binoculars_score_llama31_8b, and its weights are already "
             "in the Hub cache from that run. Uncalibrated -- the shipped "
             "constants are for Llama-3, see the llama3-8b entry.",
    ),
}

DEFAULT_PAIR = "falcon-7b"

# Every column this script writes starts with this. Same role as
# model_pairs.COLUMN_PREFIX: it is how analyse_score_human_vs_ai finds the
# columns belonging to one detector without being told each name.
COLUMN_PREFIX = "fastdetect_"


def resolve(key: str) -> Pair:
    try:
        return PAIRS[key]
    except KeyError:
        raise SystemExit(
            f"Unknown --pair {key!r}. Known pairs: {', '.join(PAIRS)}"
        ) from None


def pair_for_models(sampling: str, scoring: str) -> Pair | None:
    """The registry entry for an explicitly given pair of model ids, if any."""
    for pair in PAIRS.values():
        if pair.sampling == sampling and pair.scoring == scoring:
            return pair
    return None
