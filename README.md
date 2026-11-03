# 📊 AI Data Analyst: Healthcare Text-to-SQL Agent

An interactive, locally deployed AI agent that translates natural language questions into production-ready SQL queries. Fine-tuned on the Gemma 2B Large Language Model, this agent is specifically configured to handle complex database schemas and business logic, with a focus on medical and healthcare data environments.

## 🚀 Project Overview
This project demonstrates the end-to-end lifecycle of fine-tuning a modern Large Language Model (LLM) for a highly specific generative task (Text-to-SQL). It transitions from cloud-based GPU training on Kaggle to a lightweight, locally deployed Streamlit web application.

### Key Features
* **Custom Prompt Engineering:** Transforms raw data into a structured `<Instruction>`, `<Context>`, `<Question>`, and `<Answer>` format with strict `<EOS>` token boundaries to prevent model hallucination.
* **Balanced Dataset:** Implements strategic upsampling via `scikit-learn` to ensure the model performs equally well on simple, moderate, and challenging SQL queries.
* **Memory-Optimized Fine-Tuning:** Utilizes Low-Rank Adaptation (LoRA) and 16-bit mixed precision (`bfloat16`) to successfully fine-tune a 2-billion parameter model within strict 15GB VRAM constraints.
* **Local Web Interface:** Features a clean, interactive Streamlit frontend that caches the massive model in system RAM for fast, repeated query generation.

## 🛠️ Technology Stack
* **Model:** Google Gemma 2B (`gemma_2b_en`)
* **Machine Learning:** Keras 3, KerasHub, TensorFlow (Local Inference), JAX (Cloud Training)
* **Data Processing:** Python, Pandas, Scikit-Learn
* **Web UI:** Streamlit

## 🧠 Training Pipeline (Kaggle Cloud)
Training a 2B parameter LLM on consumer-tier GPUs requires aggressive memory management. The training script implements the following optimizations:
1. **JAX Backend:** Overrides the default TensorFlow engine to utilize JAX's superior memory recycling and prevent VRAM fragmentation during backpropagation.
2. **Mixed Precision (`bfloat16`):** Halves the model's footprint from ~9GB to ~4.5GB by explicitly setting the global Keras dtype policy before weights are loaded.
3. **Sequence Capping:** Restricts the model's attention mechanism to 512 tokens, strictly allocating memory only for the required context window.
4. **LoRA (Low-Rank Adaptation):** Freezes the base model and trains only a tiny injection of weights (Rank 4), drastically reducing computational overhead.

The final model is serialized into the official `.keras` archive format, replacing legacy `.pkl` files to ensure computational graph stability across environments.

## 💻 Local Deployment & Installation
Running modern AI models locally often introduces "Dependency Hell" due to conflicting math and serialization libraries. This project uses a strictly resolved dependency stack to ensure TensorFlow and Streamlit operate harmoniously.
