# Face Recognition with Deep Learning

This project aims to be an example of how to do face recognition using deep learning technics. The objective is, given a new person image, to find the corresponding person in the person dataset.

## How to use:
1. Import the api
2. Initialize the model with the weights
3. Fit the dataset
4. Read Image
5. Convert Image to RGB (The model is trained with RGB Images)
6. Find The match
```
from api import Face_Encode_Model
face_model = Face_Encode_Model('/home/matheus_levy/workspace/Face_Recognition/weights/vgg_embedding.keras')
face_model.fit_dataset('/home/matheus_levy/workspace/Face_Recognition/verify_dataset')
img_angela= cv2.imread('/home/matheus_levy/workspace/Face_Recognition/input_dataset/angela2.jpeg')
img_angela= cv2.cvtColor(img_angela, cv2.COLOR_BGR2RGB)
results_angela = face_model.match_image(img_angela, True, 1)
```


## Train Siamese Network Pipeline

The network used in our model is a simple VGG16 pre-trained on the dataset of VGGFACE. The Backbone is frozen and the MLP is trained. The network computes a 128 embedding vector using linear activation. This embedding represents the values that describe an image, in other words, the unique features of the face of a person in our case. 

```
vgg16 = VGG16(input_shape= input_shape, include_top= False)
vgg16.load_weights('./weights/rcmalli_vggface_tf_notop_vgg16.h5')
# FC Layer
x = Flatten()(vgg16.output)
x = Dense(512, activation='relu', name= 'fc0')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu", name= 'fc1')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu', name = 'fc2')(x)
x = BatchNormalization()(x)
x = Dense(128, activation='linear', name = 'fc3')(x)
embedding_network = Model(vgg16.input, x, name= 'Embbeding')
```

Our 128 embedding representation vector is a point in space, a point (x<sub>1</sub>, y<sub>1</sub>, z<sub>1</sub>, ...). So we can compute the similarity between embedding using square distance:
Given 2 embeddings A= (x<sub>1</sub>, y<sub>1</sub>, z<sub>1</sub>, ...) and B= (x<sub>2</sub>, y<sub>2</sub>, z<sub>2</sub>, ...) we can compute distance between this points using:

<a href="https://imgbb.com/"><img src="https://i.ibb.co/3Cc7RW8/Screenshot-4.png" alt="Point" border="0"></a>

### Triplet Loss

The triplet loss is a loss function based on distance. We have 3 points: Anchor, Positive and Negative. Anchor is an image, Positive is an image of the same class of anchor, and negative is an image of a different class than the anchor. Example: 

<a href="https://ibb.co/cDSNtxY"><img src="https://i.ibb.co/jgPLGHJ/Screenshot-5.png" alt="acn" border="0"></a>

The objective of triplet loss is to minimize the distance between the anchor and positive since they are the same class, and maximize the distance between the anchor and negative, since they are different classes.


<a href="https://ibb.co/cDSNtxY"><img src="https://user-images.githubusercontent.com/18154355/61485418-1cbb1f00-a96f-11e9-8de8-3c46eef5a7dc.png" alt="triplet" border="0"></a>

The distance is calculated in the Distance Layer:

```
anchor_positive_distance = tf.reduce_sum(tf.square(ancora - positivo), -1)
anchor_negative_distance = tf.reduce_sum(tf.square(ancora - negativo), -1)
return (anchor_positive_distance, anchor_negative_distance)
```

Example of distance calc:

<a href="https://imgbb.com/"><img src="https://i.ibb.co/mS7NTKp/triplet-loss.png" alt="triplet-loss" border="0"></a>

The final loss is calc be doing (anchor_positive_distance - anchor_negative_distance) + margin. The distance from the anchor to the positive must be lower than the anchor to the negative. The margin sets a distance difference between the two embeddings. Example:

Note that the anchor_positive_distance is lower than anchor_negative_distance. So the embedding of the anchor is closer to the positive what is good.

```
anchor_positive_distance = 0.67
anchor_negative_distance = 0.88
margin = 0.4

(anchor_positive_distance - anchor_negative_distance) + margin 
(0.67 - 0.88) + 0.4 = -0.19
```

The triplet loss is 0.19 because the distance difference from the anchor to the positive and the anchor to the negative is -0,21 and is expected to be a difference of -0.4. So if the difference is -0.4 and the margin is 0.4 we have 0 loss because anchor_positive_distance and anchor_negative_distance are in the margin. If we push the anchor_positive_distance even lower like anchor_positive_distance= 0.1 we will obtain -0.38 of loss. To prevent this from happening we apply a ReLu (in the form of tf.maximu) so the values above 0 will become 0. 
```
loss = tf.maximum((ancora_postivo_distancia - ancora_negativo_distancia) + self.margin, 0.0)
```
Representation of the learning of the Siamese model learning:
![image](https://github.com/MatheusLevy/Face-Recognition/assets/48130978/a0d4f0f9-5ff4-44ba-be55-bddffba470ba)

With this, the VGG16 network will be able to provide better embeddings. Embeddings that express unique features of the face that are different for every person.
How Siamese model trains:

![image](https://i.ibb.co/p1DsxyN/siamese-model.png)

At the end, we will have a VGG16 trained to provide 128 embedding vector that represent the unique features of a face. It's possible to compute the similarity between those vectors to know if two faces are the same or not.

### Training Note:
To minimize the effort of the model extracting the face embedding it's performed a crop of the face using the lib dlib. So the model can be trained using only the face crop.

## Face Recognition

Once we have a trained model that can provide an embedding vector we need to implement an api to use the model in a face recognition system.


### Verification Dataset
First, create a dataset with the persons that are in our verification. When receiving a new image the system will see which one of our images in the verification dataset has the closest match.
After defining the verification dataset images and persons we need to pass that to the system. That is made by calling the fit_dataset. It will extract the face, and name and compute the embeddings. These embeddings will be saved to be compared with new embeddings.

```
 def _make_embedding_dataset(self, images, names, n_max_faces):
        for image, name in zip(images, names):
            embds, _ = self.compute_embeddings_image(image, n_max_faces)
            for embd in embds:
                self.verify_embeddings.append(embd)
                self.verify_names.append(name)
        self.images = images
```

## Matching a face
Now we have an embedding dataset for all our verification images. When receiving a new image it will find a face using the face detector, compute the embedding, and compare the new embedding with the embedding dataset to find the closest match and its name. So the closest match name is the name of the new embedding.
```
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
```
