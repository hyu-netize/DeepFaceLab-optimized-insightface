from insightface.app import FaceAnalysis

class InsightFaceExtractor(object):
    def __init__(self, place_model_on_cpu=False):
        # Initialize FaceAnalysis with CUDA or CPU provider
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider' if not place_model_on_cpu else 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if not place_model_on_cpu else -1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False

    def extract(self, input_image, is_bgr=True, is_remove_intersects=False):
        # Convert BGR to RGB if needed
        if is_bgr:
            input_image = input_image[:, :, ::-1]
        
        # Detect faces using InsightFace
        faces = self.app.get(input_image)

        detected_faces = []
        for face in faces:
            l, t, r, b = [int(x) for x in face.bbox]
            if min(r - l, b - t) < 40:  # Filter small faces
                continue
            b += int((b - t) * 0.1)  # Extend bottom by 10% to cover chin
            detected_faces.append([l, t, r, b])

        # Sort faces by area
        detected_faces = [[(l, t, r, b), (r - l) * (b - t)] for (l, t, r, b) in detected_faces]
        detected_faces = sorted(detected_faces, key=lambda x: x[1], reverse=True)
        detected_faces = [x[0] for x in detected_faces]

        # Remove intersecting faces if requested
        if is_remove_intersects:
            for i in range(len(detected_faces) - 1, 0, -1):
                l1, t1, r1, b1 = detected_faces[i]
                l0, t0, r0, b0 = detected_faces[i - 1]
                dx = min(r0, r1) - max(l0, l1)
                dy = min(b0, b1) - max(t0, t1)
                if (dx >= 0) and (dy >= 0):
                    detected_faces.pop(i)

        return detected_faces