## Tasks
Go to Issues tab for details


## Setup
| Type | Recommended |
| --- | --- |
| OS | Ubuntu or WSL2 |
| CPU | 1 core |
| RAM | 4 GB |
| Disk | 30 GB |

### Main
| Setup | Don't install CPU only packages if CUDA versions were installed |
| --- | --- |
| core | sudo apt install ffmpeg python3-pip tesseract-ocr |
| base | pip3 install mediapipe tf-keras transformers torch openai-whisper pytesseract |
| extra | pip3 install datasets flax hf_xet matplotlib keras-ocr pandas scikit-learn |
| CPU | pip3 install opencv-python tensorflow |

### GPU
| Setup | This is optional and not recommended |
| --- | --- |
| Win11 | Skip to CUDA step |
| Win10 | Start VcXsrv, allow firewall, multi window, no client, no native opengl, yes disable access control |
| WSL | export GALLIUM_DRIVER=d3d12 ; sudo apt install xdg-utils mesa-utils ; glxinfo -B |
| CUDA | pip3 install jax[cuda12] opencv-cuda tensorflow[and-cuda] |


## Reference Docs
| Repo | Link |
| --- | --- |
| head-pose | https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker |
| image-text | https://tesseract-ocr.github.io/tessdoc/Installation.html |
| text-mood | https://huggingface.co/SamLowe/roberta-base-go_emotions |
| voice-text | https://github.com/openai/whisper |

![alt text](https://github.com/flowac/OnlyCode/raw/master/arch.png "arch")

