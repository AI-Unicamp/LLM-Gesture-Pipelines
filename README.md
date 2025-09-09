# From Embeddings to Language Models: A Comparative Analysis of Feature Extractors for Text-Only and Multimodal Gesture Generation

### ⚠️ Important Notice

📅 **The full implementation of the repository and pipeline configurations will be available after September 17th.**

**Johsac I. G. Sanchez, Paula D. P. Costa**
*ACM International Conference on Multimedia (ACM Multimedia) 2025*

📄 **[Paper PDF](docs/paper_ACMMM2025.pdf)** | 🌐 **[Project Page/Videos]** 

---

![Pipeline Diagram](docs/pipelines.png)
*Figure 1: Flowchart of the evaluated experimental pipelines, from our paper.*

## 📝 Abstract

Generating expressive and contextually appropriate co-speech gestures is crucial for naturalness in human-agent interaction. This study presents a systematic evaluation of seven gesture generation pipelines, comparing audio (WavLM, Whisper) and text (Word2Vec, Llama-3.2) feature extractors. We demonstrate that a smaller 3B-parameter LLM can achieve state-of-the-art performance, offering guidance for balancing generative quality with model accessibility.

## 🚀 Key Features

- 🛠️ Implementation of 7 distinct gesture generation pipelines (multimodal and text-driven).
- 📦 Pre-trained models for all evaluated pipelines, including \`Text-Only\`, \`Text-DiT\`, \`Multi-Dual\`, and more.
- 🎥 Code to run inference and generate gestures from your own audio/text files.
- 📊 Scripts for objective evaluation using metrics like FGD, BAS, DS, APSD, JM, and Dice.
- 📹 Supplementary videos showing qualitative results for all pipelines.

## 📝 Abstract

Generating expressive and contextually appropriate co-speech gestures is crucial for naturalness in human-agent interaction. This study presents a systematic evaluation of seven gesture generation pipelines, comparing audio (WavLM, Whisper) and text (Word2Vec, Llama-3.2) feature extractors. We demonstrate that a smaller 3B-parameter LLM can achieve state-of-the-art performance, offering guidance for balancing generative quality with model accessibility.

## 🚀 Key Features

- 🛠️ Implementation of 7 distinct gesture generation pipelines (multimodal and text-driven).
- 📦 Pre-trained models for all evaluated pipelines, including `Text-Only`, `Text-DiT`, `Multi-Dual`, and more.
- 🎥 Code to run inference and generate BVH gestures from audio/text inputs.
- 📊 Scripts for objective evaluation using metrics like FGD, BAS, DS, APSD, JM, and Dice.
- 📹 Supplementary videos showing qualitative results for all pipelines.

## 📂 Project Structure

📂 LLM-Gesture-Pipelines  
┣ 📜 README.md  
┣ 📜 LICENSE  
┣ 📜 environment.yml  
┣ 📜 Dockerfile  
┣ 📜 evaluation/Dockerfile  
┣ 📜 .gitignore  
┣ 📂 DiffuseStyleGesture  
┃ ┣ *(Cloned automatically in Docker build to `/root/DiffuseStyleGesture`; available as a [git submodule](https://github.com/YoungSeng/DiffuseStyleGesture.git) for local development)*  
┣ 📂 data  
┃ ┣ 📜 README.md  
┃ ┣ 📂 trn  
┃ ┣ 📂 tst  
┃ ┃ ┣ 📂 main-agent  
┃ ┃ ┃ ┣ 📂 text-audio  
┃ ┃ ┃ ┃ ┣ 📄 tst_2023_v0_028_main-agent_text_audio.npy  
┃ ┃ ┃ ┃ ┣ 📄 ...  \
┃ ┃ ┃ ┣ 📄 metadata.csv  
┃ ┣ 📄 val_2023_v0_014_main-agent.npy 
┣ 📂 models  
┃ ┣ 📂 llm  
┃ ┃ ┣ 📄 llama3b_config.py  
┃ ┣ 📄 mdm.py *(Unified model for all pipelines)*  
┃ ┣ 📂 pretrained  
┃ ┃ ┣ 📜 README.md  
┃ ┃ ┣ 📂 Basic-Whisper  
┃ ┃ ┃ ┣ 📄 model000540000.pt  
┃ ┃ ┣ 📂 Multi-DiT  
┃ ┃ ┃ ┣ 📄 model000540000.pt  
┃ ┃ ┣ 📂 Multi-Dual  
┃ ┃ ┃ ┣ 📄 model000540000.pt  
┃ ┃ ┣ 📂 Multi-Fusion  
┃ ┃ ┃ ┣ 📄 model000540000.pt  
┃ ┃ ┣ 📂 Multi-Whisper  
┃ ┃ ┃ ┣ 📄 model000540000.pt  
┃ ┃ ┣ 📂 Ref-Base  
┃ ┃ ┃ ┣ 📄 model000540000.pt  
┃ ┃ ┣ 📂 Text-DiT  
┃ ┃ ┃ ┣ 📄 model000540000.pt  
┃ ┃ ┣ 📂 Text-Only  
┃ ┃ ┃ ┣ 📄 model000540000.pt  
┣ 📂 scripts  
┃ ┣ 📄 process_embedding_training.py  
┃ ┣ 📄 train.py  
┃ ┣ 📄 inference.py  
┃ ┣ 📄 evaluate.py  
┃ ┣ 📄 model_util.py  
┣ 📂 bvh_generated  
┃ ┣ 📂 Multi-Fusion_model000540000  
┃ ┣ 📂 Multi-Dual_model000540000  
┃ ┣ 📂 ...
┣ 📂 docs  
┃ ┣ 📄 paper_ACMMM2025.pdf  
┃ ┣ 📄 pipelines.png  
┣ 📂 evaluation  
┃ ┣ 📜 environment.yml  
┃ ┣ 📜 Dockerfile  
┃ ┣ 📂 metrics  
┃ ┃ ┣ 📄 Metrics-results-generated_540k-llm.txt  
┃ ┣ 📂 videos  
┃ ┃ ┣ 📜 README.md  


## ⚙️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone --recurse-submodules https://github.com/AI-Unicamp/LLM-Gesture-Pipelines.git
   cd LLM-Gesture-Pipelines
   ```

2. **Build the Docker image for training/inference**:
   ```bash
   docker build -t llm .
   ```

3. **Build the Docker image for evaluation**:
   ```bash
   cd evaluation
   docker build -t benchmarking_sdgg_models_image .
   cd ..
   ```

4. **Run the container for training/inference**:
   ```bash
   docker run --rm -it --gpus device=0 --userns=host --shm-size 64G -v $(pwd):/workspace -p '8888:8888' --name my_container llm:latest /bin/bash
   ```
   *Note*: Scripts use `/root/DiffuseStyleGesture` by default, which is cloned automatically during the Docker build. If `/workspace/DiffuseStyleGesture` is mounted (e.g., for local development), scripts prioritize it for custom modifications.

5. **Run the container for evaluation**:
   ```bash
   docker run --rm -it --gpus device=0 --name my_evaluation_container benchmarking_sdgg_models_image:latest /bin/bash
   ```

## 🛠️ Usage

### Preprocessing
Process audio (WavLM/Whisper) and text (Llama3.2/Word2Vec) embeddings:
```bash
conda run -n llm python /workspace/scripts/process_embedding_training_v2_gg.py \
    --wavlm_path path/to/WavLM-Large.pt \
    --llm_model_path path/to/llama-3.2-3b-instruct \
    --wav_path /workspace/data/wav/ \
    --txt_path /workspace/data/tsv/ \
    --train_npy_path /workspace/output/npy/
```

### Training
Train a model (e.g., model v6: Multi-DiT):
```bash
conda run -n llm python /workspace/scripts/train.py --model /workspace/models/mdm.py --model_name Multi-DiT
```

### Inference
Generate BVH gestures from audio and text inputs for each model:
```bash
conda run -n llm python /workspace/scripts/inference.py \
    --model_name Multi-Fusion \
    --model_path /workspace/models/pretrained/Multi-Fusion/model000540000.pt \
    --txt_path /workspace/data/tst/main-agent/text-audio/ \
    --metadata_path /workspace/data/tst/

conda run -n llm python /workspace/scripts/inference.py \
    --model_name Multi-Dual \
    --model_path /workspace/models/pretrained/Multi-Dual/model000540000.pt \
    --txt_path /workspace/data/tst/main-agent/text-audio/ \
    --metadata_path /workspace/data/tst/

conda run -n llm python /workspace/scripts/inference.py \
    --model_name Text-Only \
    --model_path /workspace/models/pretrained/Text-Only/model000540000.pt \
    --txt_path /workspace/data/tst/main-agent/text-audio/ \
    --metadata_path /workspace/data/tst/

conda run -n llm python /workspace/scripts/inference.py \
    --model_name Multi-DiT \
    --model_path /workspace/models/pretrained/Multi-DiT/model000540000.pt \
    --txt_path /workspace/data/tst/main-agent/text-audio/ \
    --metadata_path /workspace/data/tst/

conda run -n llm python /workspace/scripts/inference.py \
    --model_name Text-DiT \
    --model_path /workspace/models/pretrained/Text-DiT/model000540000.pt \
    --txt_path /workspace/data/tst/main-agent/text-audio/ \
    --metadata_path /workspace/data/tst/

```
*Output*: BVH files are saved in `bvh_generated/<model_name>_model000540000/`.

### Evaluation
Compute objective metrics (FGD, BAS, DS, APSD, JM, Dice):
```bash
conda run -n sdgg python /workspace/scripts/evaluate.py
```
Results available in `evaluation/metrics/Metrics-results-generated_540k-llm.txt`.

*Note*: Use the `benchmarking_sdgg_models_image` container for evaluation, as it uses a different environment (`sdgg`) compatible with evaluation scripts.

## 📊 Metrics

Implemented metrics:
- **FGD**: Fréchet Gesture Distance
- **BAS**: Beat Alignment Score
- **DS**: Diversity Score
- **JM**: Jerk Magnitude
- **Dice**: Dice Coefficient

Detailed results in `evaluation/metrics/`.

## 📹 Videos

Evaluation videos available in `evaluation/videos/`. Follow instructions in `evaluation/videos/README.md` to download them.

## 📄 Citation

If you use this code or our models, please cite:
```bibtex
@inproceedings{sanchez2025embeddings,
  title={From Embeddings to Language Models: A Comparative Analysis of Feature Extractors for Text-Only and Multimodal Gesture Generation},
  author={Sanchez, Johsac Isbac Gomez and Costa, Paula Dornhofer Paro},
  booktitle={GENEA: Generation and Evaluation of Non-verbal Behaviour for Embodied Agents Workshop 2025},
  year={2025}
}
```

## 📚 Acknowledgments

- Based on [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture.git).
- Thanks to the libraries: PyTorch, Transformers, Librosa, and others.

## 📧 Contact

For questions or contributions, contact [paulad@unicamp.br](mailto:paulad@unicamp.br).
