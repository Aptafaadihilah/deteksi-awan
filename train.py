import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import keras_tuner as kt

# ---------------- PATH ----------------
TRAIN_DIR = "data_awan/clouds_train"
TEST_DIR = "data_awan/clouds_test"
OUTPUT_MODEL = "models/cloud_model.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# ---------------- DATA GENERATOR ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2)
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical'
)

# Test generator (tanpa augmentasi)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes
print("Kelas:", train_generator.class_indices)

# ---------------- HYPERPARAMETER TUNING ----------------
def build_model(hp):
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                             input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False  # freeze awal

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(hp.Float("dropout1", 0.2, 0.5, step=0.1))(x)
    x = layers.Dense(
        hp.Int("units", 64, 256, step=64),
        activation="relu"
    )(x)
    x = layers.Dropout(hp.Float("dropout2", 0.2, 0.5, step=0.1))(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=hp.Choice("lr", [1e-2, 1e-3, 1e-4])
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,   # bisa dinaikkan kalau mau coba lebih banyak
    directory="tuner_results",
    project_name="cloud_classification"
)

print("🔎 Mencari hyperparameter terbaik...")
tuner.search(train_generator,
             validation_data=val_generator,
             epochs=5)

best_model = tuner.get_best_models(num_models=1)[0]
print("✅ Model terbaik dari tuner sudah didapat")

# ---------------- FINE-TUNING ----------------
base_model = best_model.layers[1]  # backbone (MobileNetV2)
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

best_model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor="val_accuracy",
                             save_best_only=True, verbose=1)
early = EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True)

history = best_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early]
)

# ---------------- SIMPAN MODEL ----------------
best_model.save(OUTPUT_MODEL)
print("Model tersimpan di:", OUTPUT_MODEL)

# Simpan class indices
with open("models/class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)
print("Class indices disimpan di models/class_indices.json")

# ---------------- EVALUASI DI TEST SET ----------------
loss, acc = best_model.evaluate(test_generator)
print(f"Akurasi di test set: {acc*100:.2f}%")

import matplotlib.pyplot as plt

# -------- Plot Training History --------
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training & Validation Accuracy")

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.legend(loc="upper right")
    plt.title("Training & Validation Loss")

    plt.show()

# Panggil fungsi
plot_history(history)
