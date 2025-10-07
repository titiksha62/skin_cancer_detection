# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# import matplotlib.pyplot as plt
# import os

# # ---------------------
# # CONFIGURATION
# # ---------------------
# DATA_DIR = "melanoma_cancer_dataset/train"   # adjust if different
# LABELS = ['Benign', 'Malignant']
# IMAGE_SIZE = (128, 128)
# BATCH_SIZE = 16           # smaller batch for CPU
# EPOCHS = 5                # can increase to 10 later
# LEARNING_RATE = 1e-4
# MODEL_PATH = "skin_cancer_model.keras"

# # ---------------------
# # DATA PIPELINE
# # ---------------------
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.3  # 70% train, 30% validation
# )

# train_generator = train_datagen.flow_from_directory(
#     DATA_DIR,
#     subset="training",
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     shuffle=True,
#     seed=42
# )

# val_generator = train_datagen.flow_from_directory(
#     DATA_DIR,
#     subset="validation",
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     shuffle=False,
#     seed=42
# )

# # ---------------------
# # MODEL
# # ---------------------
# base_model = EfficientNetB0(
#     include_top=False,
#     weights="imagenet",
#     input_shape=IMAGE_SIZE + (3,)
# )
# for layer in base_model.layers[:-20]:
#     layer.trainable = False

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(256, activation="relu")(x)
# x = Dropout(0.4)(x)
# outputs = Dense(len(LABELS), activation="softmax")(x)

# model = Model(inputs=base_model.input, outputs=outputs)
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# # ---------------------
# # TRAIN
# # ---------------------
# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=EPOCHS
# )

# # ---------------------
# # SAVE MODEL
# # ---------------------
# model.save(MODEL_PATH)
# print(f"✅ Model saved as {MODEL_PATH}")

# # ---------------------
# # OPTIONAL: PLOT
# # ---------------------
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history["accuracy"], label="Train Acc")
# plt.plot(history.history["val_accuracy"], label="Val Acc")
# plt.legend()
# plt.title("Accuracy")

# plt.subplot(1, 2, 2)
# plt.plot(history.history["loss"], label="Train Loss")
# plt.plot(history.history["val_loss"], label="Val Loss")
# plt.legend()
# plt.title("Loss")
# plt.show()
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import numpy as np

# ---------------------
# CONFIGURATION
# ---------------------
DATA_DIR = "melanoma_cancer_dataset/train"  # Adjust if needed
LABELS = ['Benign', 'Malignant']
IMAGE_SIZE = (224, 224)  # EfficientNetB0 input
BATCH_SIZE = 8           # Smaller batch for CPU
EPOCHS = 25              # Slightly more epochs
LEARNING_RATE = 1e-4
MODEL_PATH = "skin_cancer_model_best.h5"   # Save best model

# ---------------------
# DATA PIPELINE
# ---------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.3  # 70% train, 30% val
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    subset="training",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    subset="validation",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=42
)

# ---------------------
# COMPUTE CLASS WEIGHTS
# ---------------------
from sklearn.utils.class_weight import compute_class_weight

labels_list = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_list),
    y=labels_list
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ---------------------
# MODEL
# ---------------------
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=IMAGE_SIZE + (3,)
)

# Freeze initial layers
for layer in base_model.layers[:-40]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)  # Slightly higher dropout
outputs = Dense(len(LABELS), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------
# CALLBACKS
# ---------------------
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=6,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ---------------------
# TRAIN
# ---------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ---------------------
# SAVE FINAL MODEL
# ---------------------
model.save("skin_cancer_model_final.h5")
print(f"✅ Model saved as skin_cancer_model_final.h5")

# ---------------------
# PLOT ACCURACY & LOSS
# ---------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()
