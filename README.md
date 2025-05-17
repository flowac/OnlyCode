## Tasks
Go to Issues tab for details


## Setup
Use Ubuntu, or WSL2  

### GPU
| Setup | This is optional |
| --- | --- |
| Win11 | Skip to CUDA step |
| Win10 | Start VcXsrv, allow firewall, multi window, no client, no native opengl, yes disable access control |
| WSL | export GALLIUM_DRIVER=d3d12 ; sudo apt install xdg-utils mesa-utils ; glxinfo -B |
| CUDA | pip3 install jax[cuda12] opencv-cuda tensorflow[and-cuda] |

### Main
| Setup | Don't install CPU only packages if CUDA versions were installed |
| --- | --- |
| core | sudo apt install ffmpeg python3-pip tesseract-ocr |
| base | pip3 install mediapipe tf-keras transformers flax hf_xet torch openai-whisper |
| extra | pip3 install datasets matplotlib keras-ocr pandas scikit-learn tqdm |
| CPU | pip3 install opencv-python tensorflow |


## Reference Docs
| Repo | Link |
| --- | --- |
| head-pose | https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker |
| image-text | https://tesseract-ocr.github.io/tessdoc/Installation.html |
| text-mood | https://huggingface.co/SamLowe/roberta-base-go_emotions |
| voice-text | https://github.com/openai/whisper |

![alt text](https://github.com/flowac/OnlyCode/raw/master/arch.png "arch")

