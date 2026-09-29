# PORT: Preference Optimization via Robust Token-Level Reweighting

**Ding Zhu, Xiukun Wei, Tian Xie, Zhihui Zhu, Xueru Zhang, Mahdi Khalili**

*NeurIPS 2026* · [Project page](https://khalilimahdi.github.io/publication/port)

PORT is a preference optimization method for aligning LLMs when preference labels are noisy. It reweights the preference objective using the empirical CDF of the model's implicit rewards, pooled over the batch, which down-weights pairs whose rewards suggest a corrupted label. The weights come from the forward pass only, so there is no extra backward pass and no external reward model.

![PORT training pipeline](https://khalilimahdi.github.io/images/publications/port-overview.png)

## Installation

```bash
git clone https://github.com/KhaliliMahdi/PORT.git
cd PORT
python -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install torch  # pick the build for your CUDA version from https://pytorch.org
pip install -r requirements.txt
```

LLaMA-2 is a gated model: accept the license on its Hugging Face page and run `huggingface-cli login` before training.

## Quick start

A tiny run that checks the full pipeline (data loading, label noise, training, evaluation, checkpointing) in a few minutes on CPU:

```bash
python train.py model=tiny-gpt2 datasets=[hh_harmless_noise] partial_ratio=0.005 \
  batch_size=4 eval_batch_size=4 gradient_accumulation_steps=1 \
  n_examples=64 n_eval_examples=16 eval_every=32 max_length=128 max_prompt_length=64 debug=true
```

Checkpoints and the resolved config are written to `.cache/<user>/<exp_name>_<timestamp>/`.

## Configuration

Training is configured with [Hydra](https://hydra.cc). Defaults are in `config/config.yaml`, and any value can be overridden on the command line.

| Group | Options | File |
|---|---|---|
| `model` | `llama2-7b`, `qwen2.5-3b`, `qwen2.5-0.5b`, `tiny-gpt2` | `config/model/*.yaml` |
| `loss` | `port`, `sft` | `config/loss/*.yaml` |

Main settings:

| Key | Default | Meaning |
|---|---|---|
| `datasets` | `[hh_harmless_noise]` | `hh_harmless_noise`, `alpaca_farm`, `hh_pairwise_noise` (all HH subsets), `hh`, `shp` |
| `noise_rate` | `0.3` | Fraction of training labels flipped (chosen ↔ rejected). Test labels are never flipped. |
| `loss.beta` | `0.1` | DPO temperature |
| `batch_size` | `64` | Total batch size; the CDF weights are computed over this batch |
| `lr` / `warmup_steps` | `5e-7` / `150` | AdamW with linear warmup |
| `trainer` | `BasicTrainer` | `BasicTrainer` (one process, model split across visible GPUs) or `FSDPTrainer` (multi-GPU data parallel) |
| `wandb.enabled` | `false` | Log to Weights & Biases (set `wandb.entity`) |

The PORT weight is `w = γ · exp(ρ · ΔCDF)` with `ρ = 0.3` and `γ = 1.5`, as in the paper. These values are set in `port_loss` in `utils/loss.py`.

## Data

All datasets download automatically from the Hugging Face Hub.

| Name | Source | Train / test pairs | Notes |
|---|---|---|---|
| `hh_harmless_noise` | [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf) `harmless-base` | 42,537 / 2,312 | Safety alignment (paper Tables 2–4) |
| `alpaca_farm` | [`tatsu-lab/alpaca_farm`](https://huggingface.co/datasets/tatsu-lab/alpaca_farm) `alpaca_human_preference.json` | 8,722 / 969 | Fixed 90/10 split (seed 42) (paper Table 1) |
| `hh_pairwise_noise` | `Anthropic/hh-rlhf`, all subsets | 160,800 / 8,552 | |

Label noise is injected at load time with probability `noise_rate`, using a fixed seed, so every method sees the same corrupted pairs.

## Reproducing the paper

As in the paper, training starts from instruction-tuned checkpoints, and the same checkpoint is used as the reference policy. No separate SFT stage is required. On 80 GB GPUs, use `trainer=FSDPTrainer` for the 7B model.

**Table 1: AlpacaFarm, noise p = 0.3**

```bash
python train.py exp_name=alpaca_llama2_port model=llama2-7b  datasets=[alpaca_farm] noise_rate=0.3 trainer=FSDPTrainer
python train.py exp_name=alpaca_qwen_port   model=qwen2.5-3b datasets=[alpaca_farm] noise_rate=0.3 trainer=FSDPTrainer
```

**Table 2: Anthropic HH (harmless), noise p = 0.3**

```bash
python train.py exp_name=hh_llama2_port model=llama2-7b datasets=[hh_harmless_noise] noise_rate=0.3 trainer=FSDPTrainer
```

**Table 3: noise intensity p ∈ {0.2, 0.3, 0.4}**

```bash
for p in 0.2 0.3 0.4; do
  python train.py exp_name=hh_llama2_port_p${p} model=llama2-7b datasets=[hh_harmless_noise] noise_rate=${p} trainer=FSDPTrainer
done
```

The final policy is saved to `<run_dir>/LATEST/policy.pt`. It is a state dict that can be loaded into the base Hugging Face model for evaluation:

```python
import torch, transformers
model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model.load_state_dict(torch.load("<run_dir>/LATEST/policy.pt", map_location="cpu")["state"])
```

### Evaluation protocol

Evaluation scripts are not included in this repository. The paper uses:

* **Gold RM score (AlpacaFarm).** Responses to the AlpacaFarm evaluation prompts are scored by a large human-preference reward model, zero-centered so the reference policy averages 0.
* **GPT-4 win rate.** Pairwise comparison against the reference (SFT) policy using the prompt in Appendix C.2 of the paper. Each pair is judged twice with the order swapped, and only consistent preferences are counted.
* **Safety rate (Anthropic HH).** The fraction of responses to PKU-Beaver prompts judged safe by the PKU-Beaver safety classifier.

## Citation

```bibtex
@inproceedings{zhu2026port,
  title     = {{PORT}: Preference Optimization via Robust Token-Level Reweighting},
  author    = {Zhu, Ding and Wei, Xiukun and Xie, Tian and Zhu, Zhihui and Zhang, Xueru and Khalili, Mohammad Mahdi},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements

This code builds on [TIS-DPO](https://github.com/exlaw/TIS-DPO) and [Direct Preference Optimization](https://github.com/eric-mitchell/direct-preference-optimization).
