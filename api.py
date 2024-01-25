class Face_Encode_Model:
    def __init__(self, model_weight_path):
        import keras
        import numpy
        import cv2
        self.model = keras.models.load_model(model_weight_path)
        self.input_shape = (150, 150)
        self.np = numpy
        self.threshold = 0.7
        self.cv2 = cv2
        self.verify_embeddings = []
        self.verify_names = []
        self.images = None

    def _normalize_img(self, img):
        img = img.astype(self.np.float32)
        img_normalized = img/255.0
        return img_normalized
    
    def _preprocessing_encoder_model(self, img):
        img = self.cv2.resize(img, self.input_shape)
        img = self._normalize_img(img)
        return img
    
    def _crop_face(self, img, faces_loc, num_crops):
        faces = []
        for i in range(num_crops):
            x, y, largura, altura = (faces_loc[i].left(), faces_loc[i].top(), faces_loc[i].width(), faces_loc[i].height())
            cropped_face = img[y:y+altura, x:x+largura]
            faces.append({'crop': cropped_face, 'bounding box': {'x': x, 'y': y, 'width': largura, 'height': altura}})
        return faces
    
    def compute_face_embeddings(self, face):
        face = self._preprocessing_encoder_model(face)
        embeddings = self.model(self.np.expand_dims(face, axis=0))
        return embeddings

    def detect_faces(self, image, max_num_faces=1):
        import dlib
        detector_faces = dlib.get_frontal_face_detector() # Uses Hog
        faces_loc = detector_faces(image)
        faces_croped = self._crop_face(image, faces_loc, num_crops=max_num_faces)
        return faces_croped

    def compute_embeddings_image(self, image, max_num_faces, face_detect=True):
        if not face_detect:
            embedd = self.compute_face_embeddings(image)
            return self.np.array(embedd), None
        
        faces = self.detect_faces(image, max_num_faces=max_num_faces)
        embeddings = []
        for face in faces:
            face_img = face['crop']
            bounding_box = face['bounding box']
            embedd = self.compute_face_embeddings(face_img)
            embeddings.append(embedd)
        return self.np.array(embeddings), bounding_box
    
    def compare_encodings(self, encoding1, enconding2):
        import keras
        cossine_similarity = keras.metrics.CosineSimilarity()
        similarity = cossine_similarity(encoding1, enconding2)
        is_same = similarity > self.threshold
        return similarity, is_same

    def _make_embedding_dataset(self, images, names, n_max_faces):
        for image, name in zip(images, names):
            embds, _ = self.compute_embeddings_image(image, n_max_faces)
            for embd in embds:
                self.verify_embeddings.append(embd)
                self.verify_names.append(name)
        self.images = images

    def fit_dataset(self, path_verify_folder):
        import os
        images, names = [], []
        folders = os.listdir(path_verify_folder)
        for folder in folders:
            path_to_folder = os.path.join(path_verify_folder, folder)
            for img_path in os.listdir(path_to_folder):
                img = self.cv2.imread(os.path.join(path_to_folder, img_path))
                img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
                images.append(img)
                names.append(folder)
        self._make_embedding_dataset(images, names, n_max_faces=1)

    def match_image(self, image, detect_faces= True, n_max_faces=1):
        new_embeddings,_ = self.compute_embeddings_image(image, n_max_faces, detect_faces)
        best_idx = None
        best_similaty = 0
        names = []
        for new_embedding in new_embeddings:
            for idx, train_embedding in enumerate(self.verify_embeddings):
                cosine_similarity, _ = self.compare_encodings(new_embedding, train_embedding)
                if cosine_similarity > best_similaty:
                    best_idx = idx
                    best_similaty = cosine_similarity
            name = self.verify_names[best_idx]
            names.append({'name': name, 'similarity': best_similaty.numpy()})
        return names

    def draw_bounding_box(self, image, bounding_box):
        x1, y1 = bounding_box['x'], bounding_box['y']
        x2, y2 = x1 + bounding_box['width'], y1 + bounding_box['height']
        self.cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return image


