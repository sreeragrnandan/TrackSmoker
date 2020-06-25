# Hi, Welcome to Track Smoker Project:

## Problem Statement:
Develop a system that can detect whether a person is smoking or not in public place and automatically report it to Law and Enforcement authorities with the location. 

### Project in a nutshell:
<ul>
<li>
Uses Machine Learning to Identify a person is smoking or not, if found smoking then report to law enforcement authority with smoker’s  picture and location
</li>
<li>
  Using Convolutional Neural Network(CNN) Architecture, Integrated with a web application for user friendliness
</li>
<li>
Main Tools Used : TensorFlow, Keras, Scikit learn, NumPy, MySQL
</li>
</ul>

### Model Structure
✌ Performed Transfer Learning with InceptionV3 and made a base model
```python
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model InceptionV3
base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
```
🤯 Constructed model, selected optimizer,loss function for solveing required problem
```python
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(2)

model2 = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

base_learning_rate = 0.0001
model2.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])
```
#### 😵 Model Summary
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
inception_v3 (Model)         (None, 8, 8, 2048)        21802784  
_________________________________________________________________
global_average_pooling2d (Gl (None, 2048)              0         
_________________________________________________________________
dense (Dense)                (None, 2)                 4098      
=================================================================
Total params: 21,806,882
Trainable params: 4,098
Non-trainable params: 21,802,784
_________________________________________________________________
```

### Data Processing 

😎 Used image augmentation as regularizer to get better generalization performance
```python
aug = ImageDataGenerator(
		rotation_range=30,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")
```
🤓 Performed Data pre-processing for inceptionV3 input
```python
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img
```
➗ Divided Training and Validation data into batches for faster training dynamics and better generalization performance, by defining a generator
🤩 At same time we Resize the image and Label the Image
```python
# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(299, 299),  # All images will be resized to 299x299
        batch_size=100,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 20 using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(299, 299),
        batch_size=40,
        class_mode='binary')
```
### 😍 Prediction Step
```python

# Smoking on validation
img=load_image("./image_name.jpg")
predict = model2.predict(np.expand_dims(img,0))
print(CLASS[np.argmax(predict)])
```
### Project Snapshot:
<img src="Architecture.gif" height="500px">
