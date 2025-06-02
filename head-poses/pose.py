import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class pose:
    pose_thres = 0.1
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    def run(self, fname):
        try:
            return self._run(fname)
        except:
            return None

    def _run(self, fname):
        image = mp.Image.create_from_file(fname)
        if image.channels < 3:
            return None
        detection_result = self.detector.detect(image)

        ret = {}
        for fbs in detection_result.face_blendshapes[0]:
            if fbs.score >= self.pose_thres:
                ret[fbs.category_name] = fbs.score
        return ret

