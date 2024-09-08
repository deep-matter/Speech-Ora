# 🎤 Real-Time Speech-to-Speech (S2S) Conversion using Hugging Face Models

**Speech-Ora** is an open-source web app for real-time speech-to-speech conversion, leveraging state-of-the-art models from the Hugging Face ecosystem to create interactions similar to GPT-like conversations.

## 📖 Table of Contents
* [Overview](#overview)
  - [System Structure](#system-structure)
  - [Modularity](#modularity)
* [Setup Instructions](#setup-instructions)
* [Usage](#usage)
  - [Server/Client Approach](#serverclient-approach)
  - [Local Approach](#local-approach)
* [Command-Line Usage](#command-line-usage)
  - [Model Parameters](#model-parameters)
  - [Generation Parameters](#generation-parameters)
  - [Key Parameters](#key-parameters)
* [Docker Setup](#docker-setup)

## Overview

### System Structure
This repository implements a speech-to-speech Server-Side/Client-side System deign with the following components based Open source model used to built so : 

1. **Voice Activity Detection (VAD)**: Powered by [Silero VAD v5](https://github.com/snakers4/silero-vad).
2. **Speech to Text (STT)**: Uses Whisper models from the Hugging Face hub.
3. **Language Model (LM)**: Any Hugging Face instruct model can be used.
4. **Text to Speech (TTS)**: Uses [Parler-TTS](https://github.com/huggingface/parler-tts) for speech synthesis.

### Modularity
The pipeline is modular and flexible, allowing customization at each stage:
- **VAD**: Integrates the [Silero VAD](https://github.com/snakers4/silero-vad).
- **STT**: Compatible with any Whisper model, including [Distil-Whisper](https://huggingface.co/distil-whisper/distil-large-v3) and multilingual variants.
- **LM**: Swap language models easily via the Hugging Face model ID.
- **TTS**: Uses Parler-TTS architecture but supports multiple checkpoints, including multilingual ones.

All components are implemented as independent classes for easy customization.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/deep-matter/Speech-Ora
   cd Speech-Ora
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The pipeline supports two primary modes of operation:
1. **Server/Client Approach**: Models run on a server, and audio input/output are streamed between client and server.
2. **Local Approach**: Uses a client-server architecture on the same machine using the loopback address.

### Server/Client Approach

1. Start the pipeline on the server:
   

2. Run the client to send audio input and receive the generated audio:
  

### Local Approach

Run the pipeline locally by using the loopback address:


## Command-Line Usage

The pipeline includes customizable parameters for running the models.

### Model Parameters

- VAD Model: The voice activity detection model.
- STT Model: Whisper model checkpoint from Hugging Face.
- LM Model: Any Hugging Face instruct model.

### Generation Parameters

- Sampling Rate: Adjustable sampling rate for audio.
- Chunk Size: Chunk size for audio processing.



## Docker Setup

To run the pipeline in a Docker container with GPU access, follow these steps:

 

### Setup the Environment Set up GPU Access for Docker

here stpes to run docker-Image of full the code run top on Containter . 
 
Before make to have Access into GPU Local host access into your container 

1. Follow the Commands line to setup your own Access Key GPU Locally to used by Docker Image 

```bash
distribution=$(. /etc/os-release;echo  $ID$VERSION_ID)  
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -  
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
2. install Nvidia-Toolkit container

```bash
sudo apt-get install -y nvidia-container-toolkit
```

3. Now, configure the Docker daemon to recognize the NVIDIA Container Runtime:

```bash 
sudo nvidia-ctk runtime configure --runtime=docker
```
Now you're ready to run the project in a container with GPU support.

### Build and Run the Docker Container




