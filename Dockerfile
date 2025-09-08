#Image that provides us with NVIDIA CUDA on Ubuntu 22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Establecer variables de entorno
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Instalar dependencias ao sistema
RUN apt-get update && apt-get install -y wget git sox libsox-fmt-all python3-distutils

# Baixar e instalar Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && mkdir /root/.conda \
    && bash /tmp/miniconda.sh -b -p /root/miniconda3 \
    && rm -f /tmp/miniconda.sh

# Definir o directorio de trabalho
WORKDIR /root

# Copiar o arquivo environment.yml ao conteiner
COPY environment.yml .

# Actualizar conda e instalar pip
RUN conda update -n base -c defaults conda -y && conda install pip -y

# Criar a ambiente conda desde environment.yml
RUN conda env create -f environment.yml

# Usar o ambiente 'llm' 
SHELL ["conda", "run", "-n", "llm", "/bin/bash", "-c"]

# Instalar dependencias adicionais
RUN pip install pydub praat-parselmouth essentia TextGrid

# Manter a terminal aberta 
CMD ["bash"]