## Status
Code samples setup. Need to start collecting data.


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
| core | sudo apt install ffmpeg python3-pip |
| base | pip3 install mediapipe datasets tf-keras transformers pandas matplotlib tqdm flax scikit-learn hf_xet torch keras-ocr openai-whisper |
| CPU | pip3 install opencv-python tensorflow |


## Reference Docs
| Repo | Link |
| --- | --- |
| head-poses | https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker |
| image-text | https://keras-ocr.readthedocs.io/en/latest/ |
| voice-mood | https://huggingface.co/SamLowe/roberta-base-go_emotions |
| voice-text | https://github.com/openai/whisper |


## Plan
Server side or in browser facial expression recognition and eye tracking  
Audio analysis (mood / emotions)  
Typing behavoir heuristics  


## Optional Objectives
CS overwatch type of review system  
