# LLM Architectures Curriculum

This module dives deep into the architecture of Large Language Models, from the original Transformer to modern variations like MoE and SSMs.

## Topics

1. **The Transformer Revolution**

   - Attention is All You Need: Scaled Dot-Product Attention, Multi-Head Attention.
   - Positional Encodings (Absolute, RoPE, ALiBi).
   - Layer Normalization (Pre-norm vs. Post-norm, RMSNorm).

2. **Efficient Attention & Scaling**

   - FlashAttention & PagedAttention.
   - KV Caching & Window Attention.
   - GQA (Grouped Query Attention) and MQA (Multi-Query Attention).

3. **Advanced Architectures**

   - Mixture of Experts (MoE): Gating mechanisms, Load balancing.
   - State Space Models (SSMs): Mamba, S4.
   - Hybrid Architectures (Jamba).

4. **Training & Fine-Tuning Paradigms**
   - Pre-training Objectives (Causal vs. Masked LM).
   - PEFT: LoRA, QLoRA, Prefix Tuning.
   - Alignment: RLHF, DPO, Kahneman-Tversky Optimization (KTO).
