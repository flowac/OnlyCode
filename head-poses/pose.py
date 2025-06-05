import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math

class pose:
    pose_thres = 0.1

    base_options = python.BaseOptions(
        model_asset_path='face_landmarker_v2_with_blendshapes.task'
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )

    detector = vision.FaceLandmarker.create_from_options(options)

    def run(self, fname):
        try:
            return self._run(fname)
        except Exception as e:
            print(f"[ERROR] Failed to process {fname}: {e}")
            return None

    def _run(self, fname):
        image = mp.Image.create_from_file(fname)
        if image.channels < 3:
            return None
        detection_result = self.detector.detect(image)

        ret = {}

        # Extract blendshapes
        for fbs in detection_result.face_blendshapes[0]:
            if fbs.score >= self.pose_thres:
                ret[fbs.category_name] = fbs.score

        # Extract transformation matrix and compute yaw/pitch/roll
        if detection_result.facial_transformation_matrixes:
            matrix = detection_result.facial_transformation_matrixes[0].data
            mat = np.array(matrix).reshape((4, 4))
            rotation = mat[:3, :3]

            angles = self.rotation_matrix_to_euler_angles(rotation)
            ret['head_pose_angles'] = angles

        return ret

    def rotation_matrix_to_euler_angles(self, R):
        """
        Converts a 3x3 rotation matrix to yaw, pitch, and roll in degrees.
        Assumes Tait–Bryan angles (YXZ order).
        """
        sy = math.sqrt(R[0, 0] ** 2 + R[2, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            yaw = math.atan2(R[1, 0], R[1, 1])
            pitch = math.atan2(-R[1, 2], sy)
            roll = math.atan2(R[0, 2], R[2, 2])
        else:
            yaw = math.atan2(-R[2, 1], R[2, 2])
            pitch = math.atan2(-R[1, 2], sy)
            roll = 0

        return {
            'yaw': math.degrees(yaw),
            'pitch': math.degrees(pitch),
            'roll': math.degrees(roll)
        }
