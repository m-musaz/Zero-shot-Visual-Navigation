# Thinking in 360°: Zero-Shot Vision-and-Language Navigation

A browser-based demo is available for visualizing navigation behavior and agent reasoning:

🔗 https://nav-vision-sproj.vercel.app

This repository contains the implementation and system design for **Thinking in 360°**, a zero-shot Vision-and-Language Navigation (VLN) framework that enables an embodied agent to follow natural language instructions in **photorealistic indoor environments** without finetuning or task-specific visual training.

The project is built and evaluated on the **Room-to-Room (R2R)** benchmark using the **Matterport3D** simulator.

---

## 🚀 Project Overview

Vision-and-Language Navigation (VLN) requires an agent to interpret free-form natural language instructions (e.g., *“Walk down the hall and stop near the sink”*) and navigate complex indoor environments.

Unlike prior VLN approaches that rely on heavy supervision, pretrained visual backbones, or finetuned policies, this project explores a **training-free, zero-shot approach** by framing navigation as a **reasoning problem handled by large language models (LLMs)**.

---

## ✨ Key Contributions

- **Zero-shot VLN pipeline** (no finetuning, no pretrained visual navigation models)
- **360° panoramic scene understanding** with efficient view filtering
- **Graph-based spatial memory** for long-horizon reasoning
- **Lookahead exploration** to reason about future viewpoints before acting
- **Reasoning-centric prompting** with explicit step justifications
- **Cycle detection and recovery** to prevent navigation loops
- **Goal detection via semantic matching** instead of oracle signals
- Competitive performance on **unseen R2R environments**

---

## 🧠 System Architecture

The system is split into two decoupled components:

- **Simulation Module (`driver.py`)**
  - Runs inside the Matterport3D simulator
  - Captures panoramic views and executes navigation actions

- **Inference Server (`server.py`)**
  - Flask-based microservice running on GPU
  - Handles image captioning, LLM reasoning, spatial graph updates, and decision-making

Communication between the simulator and inference server is handled via REST APIs.

---

## 🖼️ Perception and Reasoning

- Panoramic views are captured at each step and filtered to left / forward / right perspectives
- Images are converted into **navigational scene descriptions** using a vision-language model
- A **dynamic graph** stores visited viewpoints, connections, and summaries
- The LLM selects the next action by reasoning over:
  - Current scene description
  - Navigation history
  - Spatial graph context
  - Instruction goal

---

## 📊 Evaluation

- **Dataset:** Room-to-Room (R2R), validation unseen split  
- **Metrics:** SR, SPL, NE, OSR, Trajectory Length  
- **Setting:** Fully zero-shot (no R2R finetuning)

The method achieves competitive performance compared to other zero-shot and language-only VLN approaches while remaining training-free and interpretable.

---

## 🌐 Interactive Demo

A browser-based demo is available for visualizing navigation behavior and agent reasoning:

🔗 https://nav-vision-sproj.vercel.app

The demo allows interactive exploration of Matterport3D scans and replay of navigation trajectories.

---

## 📦 Reproducibility

The repository includes:
- Docker setup for the Matterport3D simulator
- Flask inference server
- Evaluation scripts compatible with the official R2R metrics

Detailed setup instructions are provided in the project documentation.

---

## 📌 Citation

If you use this work, please cite the associated undergraduate thesis:

**Thinking in 360°: A Zero-Shot Navigation Agent for Photorealistic Indoor Spaces**  
Hasan Hameed, Ahmad Faraz, Muhammad Musa Zulfiqar  
Supervisor: Dr. Muhammad Tahir, LUMS

---

## 📜 License

This project is released for academic and research use.
