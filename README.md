# From Embeddings to Language Models: A Comparative Analysis of Feature Extractors for Text-Only and Multimodal Gesture Generation

### ⚠️ Important Notice

📅 **The full implementation of the repository and pipeline configurations will be available after September 24th.**

**Johsac I. G. Sanchez, Paula D. P. Costa**
*ACM International Conference on Multimedia (ACM Multimedia) 2025*

📄 **[Paper PDF](https://genea-workshop.github.io/2025/workshop/#accepted-papers)** | 🌐 **[Project Page/Videos]** 

---

![Pipeline Diagram](docs/pipelines.png)
*Figure 1: Flowchart of the evaluated experimental pipelines, from our paper.*

## 📝 Abstract

Generating expressive and contextually appropriate co-speech gestures is crucial for naturalness in human-agent interaction. This study presents a systematic evaluation of seven gesture generation pipelines, comparing audio (WavLM, Whisper) and text (Word2Vec, Llama-3.2) feature extractors. We demonstrate that a smaller 3B-parameter LLM can achieve state-of-the-art performance, offering guidance for balancing generative quality with model accessibility.

## 🚀 Key Features

- 🛠️ Implementation of 7 distinct gesture generation pipelines (multimodal and text-driven).
- 📦 Pre-trained models for all evaluated pipelines, including \`Text-Only\`, \`Text-DiT\`, \`Multi-Dual\`, and more.
- 🎥 Code to run inference and generate gestures from audio/text files.
- 📊 Scripts for objective evaluation using metrics like FGD, BAS, DS, JM, and GAC Dice.
- 📹 Supplementary videos showing qualitative results for all pipelines.

## 📂 Project Structure

📂 LLM-Gesture-Pipelines  
┣ 📜 README.md  
┣ 📜 LICENSE  
┣ 📜 environment.yml  
┣ 📜 Dockerfile  \
┣ 📜 .gitignore  
┣ 📂 DiffuseStyleGesture  
┃ ┣ *(Cloned automatically in Docker build to `/root/DiffuseStyleGesture`; available as a [git submodule](https://github.com/YoungSeng/DiffuseStyleGesture.git) for local development)*  
┣ 📂 data  
┃ ┣ 📜 README.md  
┃ ┣ 📂 trn  
┃ ┣ 📂 tst  
┃ ┃ ┣ 📂 main-agent  \ 
┃ ┃ ┃ ┣ 📂 wav \ 
┃ ┃ ┃ ┣ 📂 tsv \ 
┃ ┃ ┃ ┣ 📂 text-audio  
┃ ┃ ┃ ┃ ┣ 📄 tst_2023_v0_028_main-agent_text_audio.npy  
┃ ┃ ┃ ┃ ┣ 📄 ...  \ 
┃ ┃ ┃ ┣ 📄 metadata.csv  
┃ ┣ 📄 val_2023_v0_014_main-agent.npy \ 
┣ 📂 models  \ 
┃ ┣ 📂 WavLM \ 
┃ ┃ ┗ 📄 WavLM-Large.pt *(Manual download required)* \ 
┃ ┣ 📂 llama-3.2-3b-instruct  *(Manual download required)*  \ 
┃ ┃ ┣ 📄 config.json  \ 
┃ ┃ ┣ 📄 model-00001-of-00002.safetensors  \ 
┃ ┃ ┣ 📄 model-00002-of-00002.safetensors  \ 
┃ ┃ ┣ 📄 tokenizer.json  \ 
┃ ┃ ┣ *(Other configuration and model files)* \ 
┃ ┣ 📂 pretrained  
┃ ┃ ┣ 📜 README.md  
┃ ┃ ┣ 📂 Basic-Whisper  
┃ ┃ ┃ ┣ 📄 model000540000.pt  
┃ ┃ ┣ 📂 Multi-DiT  
┃ ┃ ┃ ┣ 📄 model000540000.pt  
┃ ┃ ┣ 📂 ... \ 
┃ ┣ 📄 mdm.py *(Unified model for all pipelines)*  \ 
┃ ┣ 📜 DiffuseStyleGesture.yml *(configuration for all pipelines)*  \ 
┣ 📂 scripts  
┃ ┣ 📄 process_embedding.py  
┃ ┣ 📄 train.py  
┃ ┣ 📄 inference.py  
┃ ┣ 📄 evaluate.py  
┃ ┣ 📄 model_util.py  
┣ 📂 bvh_generated  
┃ ┣ 📂 Multi-Fusion_model000540000  
┃ ┣ 📂 Multi-Dual_model000540000  
┃ ┣ 📂 ... \
┣ 📂 docs  
┃ ┣ 📄 pipelines.png \
┣ 📂 evaluation  
┃ ┣ 📜 environment.yml  
┃ ┣ 📜 Dockerfile  
┃ ┣ 📂 metrics  
┃ ┃ ┣ 📄 Metrics-results-generated_540k-llm.txt  
┃ ┣ 📂 videos  
┃ ┃ ┣ 📜 README.md  \
┣ 📂 examples  
┃ ┣ 📄 generate_gestures.py  
┃ ┣ 📂 sample_input  
┃ ┃ ┣ 📄 sample.txt  
┃ ┃ ┣ 📄 sample.wav  
┃ ┣ 📄 sample_output.bvh  


## ⚙️ Setup & Installation

### 📝 Hardware Note

This project was developed and tested on a workstation with the following configuration:

* **GPU**: NVIDIA Quadro RTX 5000 (16 GB VRAM)

A GPU with at least **16 GB VRAM** is recommended to comfortably run the processing and training scripts, as large models such as Llama 3 (3B) and WavLM-Large are loaded simultaneously.

1. **Clone the repository**:
   ```bash
   git clone --recurse-submodules https://github.com/AI-Unicamp/LLM-Gesture-Pipelines.git
   cd LLM-Gesture-Pipelines
   ```

2. **Build the Docker image for training/inference**:
   ```bash
   docker build -t llm .
   ```

3. **Download Pre-trained Models**:

   Before running the processing or training, you must download the following models and place them in the specified paths:

   * **WavLM-Large**: Used for audio feature extraction.
     * **🔗 Download Link**: [Microsoft WavLM-Large.pt](https://github.com/microsoft/unilm/blob/master/wavlm/README.md) (Look for the direct link to the `WavLM-Large.pt` checkpoint).
     * **📍 Required Location**: `models/WavLM/WavLM-Large.pt`.

   * **Llama 3.2 (3B)**: Used for text feature extraction.
     * **🔗 Download Link**: [Hugging Face Llama 3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) (Requires authentication; follow instructions to download model weights).
     * **📍 Required Location**: `models/llama-3.2-3b-instruct/` (includes `config.json`, `model-*.safetensors`, `tokenizer.json`, etc.).


4. **Run the container for training/inference**:
   ```bash
   docker run --rm -it --gpus device=0 --userns=host --shm-size 64G -v $(pwd):/workspace -p '8888:8888' --name my_container llm:latest /bin/bash
   ```
   *Note*: Scripts use `/root/DiffuseStyleGesture` by default, which is cloned automatically during the Docker build. If `/workspace/DiffuseStyleGesture` is mounted (e.g., for local development), scripts prioritize it for custom modifications.

5. **Change to the project directory and activate the Conda environment:**:
```bash
cd /workspace
source activate llm
```

## 🛠️ Usage

### Preprocessing

This section explains how to preprocess audio and text data to generate combined embeddings for gesture generation. The script `process_embedding.py` processes `.wav` files (audio) and `.tsv` files (text with timestamps) to produce `.npy` files containing concatenated audio and text embeddings, which are used as input for inference.

#### Step-by-Step Guide

1. **Prepare Input Data**:
   - **Audio Files**: Place `.wav` files in `data/tst/main-agent/wav/` (e.g., `tst_2023_v0_000_main-agent.wav`). These should be mono audio files with a sample rate of 44100 Hz for optimal compatibility.
   - **Text Files**: Place `.tsv` files in `data/tst/main-agent/tsv/` (e.g., `tst_2023_v0_000_main-agent.tsv`). Each `.tsv` file should contain three columns: `start_time`, `end_time`, and `word`, with timestamps in seconds.

2. **Run the Preprocessing Script**:
   - Inside the Docker container, activate the Conda environment and run:
     ```bash
     python scripts/process_embedding.py \
         --wav_path data/tst/main-agent/wav/ \
         --txt_path data/tst/main-agent/tsv/ \
         --tst_npy_path data/tst/main-agent/text-audio/ \
         --process_tst \
     ```
   - **Arguments**:
     - `--wavlm_path` and `--llm_model_path`: Paths to the pre-trained WavLM and Llama 3.2 models.
     - `--wav_path` and `--txt_path`: Directories containing `.wav` and `.tsv` files.
     - `--tst_npy_path`: Output directory for `.npy` files (e.g., `data/tst/main-agent/text-audio/`).
     - `--process_tst`: Process the test split (use `--process_train` or `--process_val` for other splits in the future).
     - `--limit_files`: Optional, limits the number of files processed (e.g., `2` for testing).

3. **Output**:
   - The script generates `.npy` files in `data/tst/main-agent/text-audio/`.
   - Each `.npy` file contains concatenated embeddings:
     - **Audio Embeddings**: From WavLM-Large (dimension: 1133, including MFCC, spectrum, prosody, WavLM features, and onset).
     - **Text Embeddings**: From Llama 3.2 (dimension: 3074, including 3072-dimensional embeddings plus laugh and onset features).




## 🏃 Running Inference
To generate BVH gestures from audio and text inputs, first navigate to the project's root directory (`/workspace`) inside the container. This ensures that Python can correctly find the necessary modules.

1. **Use in Inference**:
   - The generated `.npy` files are used as input for the inference script (`inference.py`). For example:
     ```bash
     python scripts/inference.py \
         --model_name Multi-Fusion \
         --model_path models/pretrained/Multi-Fusion/model000540000.pt \
         --txt_path data/tst/main-agent/text-audio/ \
         --metadata_path data/tst/
     ```


1. **Generate gestures for each model:**
Generate BVH gestures from the audio and text `.npy` files for each model:
```bash
# Multi-Fusion
python scripts/inference.py \
    --model_name Multi-Fusion \
    --model_path models/pretrained/Multi-Fusion/model000540000.pt \
    --txt_path data/tst/main-agent/text-audio/ \
    --metadata_path data/tst/

# Multi-Dual
python scripts/inference.py \
    --model_name Multi-Dual \
    --model_path models/pretrained/Multi-Dual/model000540000.pt \
    --txt_path data/tst/main-agent/text-audio/ \
    --metadata_path data/tst/

# Text-Only
python scripts/inference.py \
    --model_name Text-Only \
    --model_path models/pretrained/Text-Only/model000540000.pt \
    --txt_path data/tst/main-agent/text-audio/ \
    --metadata_path data/tst/

# Multi-DiT
python scripts/inference.py \
    --model_name Multi-DiT \
    --model_path models/pretrained/Multi-DiT/model000540000.pt \
    --txt_path data/tst/main-agent/text-audio/ \
    --metadata_path data/tst/

# Text-DiT
python scripts/inference.py \
    --model_name Text-DiT \
    --model_path models/pretrained/Text-DiT/model000540000.pt \
    --txt_path data/tst/main-agent/text-audio/ \
    --metadata_path data/tst/

```
*Output*: BVH files are saved in `bvh_generated/<model_name>_model000540000/`.


### Training
Train a model (e.g., model v6: Multi-DiT):
```bash
conda run -n llm python /workspace/scripts/train.py --model /workspace/models/mdm.py --model_name Multi-DiT
```

### Evaluation

4. **Build the Docker image for evaluation**:
   ```bash
   cd evaluation
   docker build -t benchmarking_sdgg_models_image .
   cd ..
   ```


6. **Run the container for evaluation**:
   ```bash
   docker run --rm -it --gpus device=0 --name my_evaluation_container benchmarking_sdgg_models_image:latest /bin/bash
   ```


Compute objective metrics (FGD, BAS, DS, JM, GAC Dice):
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
- **GAC Dice**: Dice Coefficient

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