# Using Sesame TTS in AI-Slop-Master

This document explains how to set up and use Sesame TTS in the AI-Slop-Master project.

## Prerequisites

- Python 3.10 or later
- Git
- Access to the SesameAILabs/csm repository
- Hugging Face account with access to CSM-1B and Llama-3.2-1B models
- PyTorch and torchaudio

## Setup

1. Run the setup script to install Sesame TTS:

```bash
python setup_sesame.py
```

2. Log in to Hugging Face:

```bash
huggingface-cli login
```

3. Make sure you have access to the required models:
   - CSM-1B
   - Llama-3.2-1B

4. Before running the application, set the environment variable to disable torch compilation:

```bash
# On Linux/Mac
export NO_TORCH_COMPILE=1

# On Windows
set NO_TORCH_COMPILE=1
```

## How Sesame TTS Works

Sesame uses speaker IDs (integers) instead of voice names. In our application:

- Speaker ID 0: Used for male voices (Narrator, AI Debater 2, Valentino)
- Speaker ID 1: Used for female voices (AI Debater 1, Jane)

The first time you run synthesis, models will be downloaded from Hugging Face.

## Running the Application

To run the application with Sesame TTS properly configured, use the provided wrapper script:

```bash
python run_with_sesame.py
```

Or to run a specific script:

```bash
python run_with_sesame.py your_script.py
```

## Troubleshooting

- If you encounter errors about missing modules or models, make sure:
  1. You've run `setup_sesame.py`
  2. You're logged in to Hugging Face
  3. You have access to the required models

- If you get errors about torch compilation, make sure:
  1. The environment variable `NO_TORCH_COMPILE=1` is set
  2. You're using the `run_with_sesame.py` script which sets this for you
