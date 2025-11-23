# Octavia
![LunarTech Logo](documentation/assets/lunartech_logo.png)

**Rise Beyond Language**

![Octavia Dashboard](documentation/assets/dashboard_preview.png)

## 🌌 Overview

**Octavia** is a next-generation, cloud-native video translation platform designed to break down language barriers with cinematic quality. Built with a "Liquid Glass" aesthetic, it combines powerful AI models with a stunning, intuitive user interface.

Unlike traditional tools that simply overlay text, Octavia preserves the original experience. Our **Magic Mode** separates vocals from background music, translates the speech with context-aware LLMs, clones the original speaker's voice, and seamlessly remixes everything back together—all while keeping the original background audio intact.

## ✨ Key Features

*   **🔮 Magic Mode:** The heart of Octavia. It uses a sophisticated pipeline to:
    *   **Isolate Vocals:** Removes speech while keeping music and SFX (via UVR5).
    *   **Clone Voices:** Generates dubbed audio that sounds like the original speaker (via Coqui XTTS v2).
    *   **Sync Audio:** Automatically adjusts timing to match the video's lip movements.
*   **🌊 Liquid Glass Design:** A premium, dark-mode UI featuring glassmorphism, deep purple gradients, and subtle glow effects that feel alive.
*   **⚡ Cloud-Native Power:** Powered by a serverless GPU fleet (RunPod) to handle heavy AI processing without slowing down your device.
*   **🧠 Context-Aware Translation:** Uses advanced LLMs (GPT-4, Claude, Llama 3) to understand the nuance of the entire video, ensuring translations are accurate and culturally relevant.
*   **🎯 Precision Editing:** Full control over subtitles, timing, and audio mixing.

## 🛠️ Tech Stack

Octavia is built on a modern, scalable stack:

*   **Frontend:** Next.js 15 (App Router), Tailwind CSS, shadcn/ui, Framer Motion.
*   **Backend:** FastAPI (Python), Celery, Redis.
*   **AI & Compute:** RunPod (Serverless GPUs), PyTorch, FFmpeg.
*   **Models:** WhisperX (Transcription), Coqui XTTS v2 (Voice Cloning), UVR5 (Vocal Separation).
*   **Database:** Neon (Serverless Postgres).

## 📚 Documentation

Explore our detailed documentation to understand how Octavia works:

*   **[User Flow](documentation/connections/user_flow.md):** Visual guide to the application's navigation and page connections.
*   **[Production Architecture](documentation/production_architecture.md):** Deep dive into the cloud-native system design.
*   **[PyVideoTrans Explained](documentation/pyvideotrans_explained.md):** Detailed breakdown of the core video translation workflow.
*   **[Recommended Tools](documentation/recommended_tools.md):** The best open-source tools for building video AI pipelines.
*   **[Translation Strategies](documentation/translation_accuracy_strategies.md):** How we handle duration mismatches and ensure accuracy.

## 🚀 Getting Started

*(Coming Soon: Instructions for local setup and deployment)*

---

*Octavia is currently under active development. Join us in shaping the future of global communication.*

---

## 🌐 Connect with LunarTech

*   **Website:** [lunartech.ai](http://lunartech.ai/)
*   **LinkedIn:** [LunarTech AI](https://www.linkedin.com/company/lunartechai)
*   **Instagram:** [@lunartech.ai](https://www.instagram.com/lunartech.ai/)
*   **Substack:** [LunarTech on Substack](https://substack.com/@lunartech)

## 📧 Contact

*   **Tatev:** [tatev@lunartech.ai](mailto:tatev@lunartech.ai)
*   **Vahe:** [vahe@lunartech.ai](mailto:vahe@lunartech.ai)
*   **Open Source:** [opensource@lunartech.ai](mailto:opensource@lunartech.ai)
