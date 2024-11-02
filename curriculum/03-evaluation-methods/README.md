# Evaluation Methods Curriculum

This module focuses on how to measure the performance, reliability, and safety of LLMs and AI systems.

## Topics

1. **Standard LLM Benchmarks**

   - General Knowledge: MMLU (Massive Multitask Language Understanding).
   - Reasoning: GSM8K (Grade School Math 8K), HumanEval (Coding).
   - Frameworks: HELM (Holistic Evaluation of Language Models).

2. **RAG Evaluation (The RAG Triad)**

   - Faithfulness: Is the answer grounded in the context?
   - Answer Relevance: Does the answer address the query?
   - Context Precision: Is the retrieved context relevant to the query?
   - Tools: Ragas, TruLens, Arize Phoenix.

3. **Human-in-the-Loop & Model-as-a-Judge**

   - Side-by-side comparison (LMSYS Chatbot Arena methodology).
   - Using GPT-4 or Claude-3 as a judge for qualitative metrics.
   - LLM-eval-LLM patterns and pitfalls.

4. **Safety & Red Teaming**
   - Jailbreaking techniques and mitigations.
   - PII detection and redactions.
   - Toxicity and bias auditing.
   - Adversarial testing for prompt injections.
