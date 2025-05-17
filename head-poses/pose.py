import sys, time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

flist = ["image.png"]
pose_thres = 0.1

if len(sys.argv) > 1:
    flist = sys.argv[1:]

def pose(fname):
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    start_time = time.time()
    image = mp.Image.create_from_file(fname)
    detection_result = detector.detect(image)
    stop_time = time.time()

    for fbs in detection_result.face_blendshapes[0]:
        if fbs.score >= pose_thres:
            print(f'{fbs.category_name: <20} {fbs.score}')
    print(f'Pose detect took {(stop_time - start_time) * 1000} ms')

for fname in flist:
    pose(fname)

