# From Embeddings to Language Models: A Comparative Analysis of Feature Extractors for Text-Only and Multimodal Gesture Generation

📅 **Final submission is expected before September 12.**

**Johsac I. G. Sanchez, Paula D. P. Costa**
*ACM International Conference on Multimedia (ACM Multimedia) 2025*

**[Paper PDF]** | **[Project Page/Videos]** 

---

![Pipeline Diagram](URL_A_TU_FIGURA_1.png)
*Figure 1: Flowchart of the evaluated experimental pipelines, from our paper.*

## Abstract
Generating expressive and contextually appropriate co-speech gestures is crucial for naturalness in human-agent interaction. This study presents a systematic evaluation of seven gesture generation pipelines, comparing audio (WavLM, Whisper) and text (Word2Vec, Llama-3.2) feature extractors. We demonstrate that a smaller 3B-parameter LLM can achieve state-of-the-art performance, offering guidance for balancing generative quality with model accessibility.

## 🚀 Key Features
- Implementation of 7 distinct gesture generation pipelines (multimodal and text-driven).
- Pre-trained models for all evaluated pipelines, including `Text-Only`, `Text-DiT`, `Multi-Dual`, ....
- Code to run inference and generate gestures from your own audio/text files.
- Scripts for objective evaluation using metrics like FGD, BAS, and Jerk.
- Supplementary videos showing qualitative results for all pipelines.

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AI-Unicamp/LLM-Gesture-Pipelines.git](https://github.com/AI-Unicamp/LLM-Gesture-Pipelines.git)
    cd LLM-Gesture-Pipelines
    ```
