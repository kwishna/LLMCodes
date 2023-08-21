"""
Original file is located at - https://colab.research.google.com/drive/1-lJP8fRwaJbSFy3xe1YH6NLgMj1xhWwd
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content
!git clone https://github.com/lllyasviel/Fooocus

!apt -y update -qq
!wget https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O /content/libtcmalloc_minimal.so.4
# %env LD_PRELOAD=/content/libtcmalloc_minimal.so.4

!pip install torchsde==0.2.5 einops==0.4.1 transformers==4.30.2 safetensors==0.3.1 accelerate==0.21.0
!pip install pytorch_lightning==1.9.4 omegaconf==2.2.3 gradio==3.39.0 xformers==0.0.20 triton==2.0.0 pygit2==1.12.2

!apt -y install -qq aria2
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/sd_xl_base_1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors -d /content/Fooocus/models/checkpoints -o sd_xl_base_1.0_0.9vae.safetensors
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/sd_xl_refiner_1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors -d /content/Fooocus/models/checkpoints -o sd_xl_refiner_1.0_0.9vae.safetensors
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors -d /content/Fooocus/models/loras -o sd_xl_offset_example-lora_1.0.safetensors

# %cd /content/Fooocus
!git pull
!python launch.py --share