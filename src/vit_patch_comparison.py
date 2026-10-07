import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

IMAGE_SIZE = 32
NUM_CLASSES = 10

PATCH_SIZES = [4, 8, 16]

PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 4

MLP_HEAD_UNITS = [128, 64]

BATCH_SIZE = 128
EPOCHS = 5


# ============================================================
# 2. LOAD CIFAR-10 DATASET
# ============================================================

print("=" * 60)
print("Loading CIFAR-10 dataset...")
print("=" * 60)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

print("Training images:", x_train.shape)
print("Testing images :", x_test.shape)


# CIFAR-10 class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ============================================================
# 3. CREATE RESULTS FOLDER
# ============================================================

os.makedirs("results", exist_ok=True)


# ============================================================
# 4. VISION TRANSFORMER COMPONENTS
# ============================================================

class Patches(layers.Layer):
    """
    Divides an image into non-overlapping patches.
    """

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[
                1,
                self.patch_size,
                self.patch_size,
                1
            ],
            strides=[
                1,
                self.patch_size,
                self.patch_size,
                1
            ],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )

        patch_dims = patches.shape[-1]

        patches = tf.reshape(
            patches,
            [
                batch_size,
                -1,
                patch_dims
            ]
        )

        return patches


class PatchEncoder(layers.Layer):
    """
    Projects patches into embedding vectors and adds
    positional information.
    """

    def __init__(self, num_patches, projection_dim):
        super().__init__()

        self.num_patches = num_patches

        self.projection = layers.Dense(
            units=projection_dim
        )

        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1,
            output_dim=projection_dim
        )

        self.class_token = self.add_weight(
            name="class_token",
            shape=(1, 1, projection_dim),
            initializer="zeros",
            trainable=True
        )

    def call(self, patches):

        batch_size = tf.shape(patches)[0]

        projected_patches = self.projection(patches)

        class_token = tf.broadcast_to(
            self.class_token,
            [
                batch_size,
                1,
                tf.shape(projected_patches)[-1]
            ]
        )

        encoded = tf.concat(
            [class_token, projected_patches],
            axis=1
        )

        positions = tf.range(
            start=0,
            limit=self.num_patches + 1,
            delta=1
        )

        encoded = encoded + self.position_embedding(positions)

        return encoded


# ============================================================
# 5. CREATE VISION TRANSFORMER
# ============================================================

def create_vit_classifier(patch_size):

    num_patches = (
        IMAGE_SIZE // patch_size
    ) ** 2

    inputs = layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # Normalize pixel values
    normalized = layers.Rescaling(
        1.0 / 255
    )(inputs)

    # Convert image into patches
    patches = Patches(
        patch_size
    )(normalized)

    # Encode patches
    encoded_patches = PatchEncoder(
        num_patches,
        PROJECTION_DIM
    )(patches)

    # Transformer blocks
    for _ in range(TRANSFORMER_LAYERS):

        # Layer normalization
        x1 = layers.LayerNormalization(
            epsilon=1e-6
        )(encoded_patches)

        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=PROJECTION_DIM // NUM_HEADS,
            dropout=0.1
        )(
            x1,
            x1
        )

        # Skip connection
        x2 = layers.Add()(
            [attention_output, encoded_patches]
        )

        # Layer normalization
        x3 = layers.LayerNormalization(
            epsilon=1e-6
        )(x2)

        # MLP
        x3 = layers.Dense(
            PROJECTION_DIM * 2,
            activation=tf.nn.gelu
        )(x3)

        x3 = layers.Dropout(0.1)(x3)

        x3 = layers.Dense(
            PROJECTION_DIM
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()(
            [x3, x2]
        )

    # Classification token
    representation = layers.LayerNormalization(
        epsilon=1e-6
    )(encoded_patches[:, 0])

    # Classification head
    features = layers.Dense(
        MLP_HEAD_UNITS[0],
        activation="gelu"
    )(representation)

    features = layers.Dropout(0.3)(features)

    features = layers.Dense(
        MLP_HEAD_UNITS[1],
        activation="gelu"
    )(features)

    features = layers.Dropout(0.3)(features)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(features)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"ViT_Patch_{patch_size}"
    )

    return model


# ============================================================
# 6. DISPLAY PATCH INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("PATCH INFORMATION")
print("=" * 60)

for patch_size in PATCH_SIZES:

    num_patches = (
        IMAGE_SIZE // patch_size
    ) ** 2

    print(
        f"Patch size: {patch_size}x{patch_size} "
        f"--> Number of patches: {num_patches}"
    )


# ============================================================
# 7. TRAIN MODELS FOR DIFFERENT PATCH SIZES
# ============================================================

results = []

histories = {}

models = {}


for patch_size in PATCH_SIZES:

    print("\n")
    print("=" * 60)
    print(
        f"TRAINING VISION TRANSFORMER "
        f"WITH {patch_size}x{patch_size} PATCHES"
    )
    print("=" * 60)

    num_patches = (
        IMAGE_SIZE // patch_size
    ) ** 2

    model = create_vit_classifier(
        patch_size
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    start_time = time.time()

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        verbose=1
    )

    training_time = (
        time.time() - start_time
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    results.append({
        "Patch Size": f"{patch_size}x{patch_size}",
        "Number of Patches": num_patches,
        "Parameters": model.count_params(),
        "Test Accuracy": test_accuracy,
        "Training Time (seconds)": training_time
    })

    histories[patch_size] = history
    models[patch_size] = model

    print(
        f"\nPatch Size: {patch_size}x{patch_size}"
    )

    print(
        f"Number of Patches: {num_patches}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Training Time: "
        f"{training_time:.2f} seconds"
    )


# ============================================================
# 8. CREATE COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    "results/vit_patch_comparison.csv",
    index=False
)


# ============================================================
# 9. PLOT VALIDATION ACCURACY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for patch_size in PATCH_SIZES:

    history = histories[patch_size]

    plt.plot(
        history.history["val_accuracy"],
        marker="o",
        label=f"{patch_size}x{patch_size}"
    )

plt.title(
    "Validation Accuracy for Different Patch Sizes"
)

plt.xlabel("Epoch")

plt.ylabel("Validation Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/validation_accuracy_comparison.png"
)

plt.show()


# ============================================================
# 10. PLOT TRAINING LOSS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for patch_size in PATCH_SIZES:

    history = histories[patch_size]

    plt.plot(
        history.history["val_loss"],
        marker="o",
        label=f"{patch_size}x{patch_size}"
    )

plt.title(
    "Validation Loss for Different Patch Sizes"
)

plt.xlabel("Epoch")

plt.ylabel("Validation Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/validation_loss_comparison.png"
)

plt.show()


# ============================================================
# 11. PATCH REPRESENTATION VISUALIZATION
# ============================================================

sample_image = x_test[0]

plt.figure(
    figsize=(12, 4)
)

for index, patch_size in enumerate(PATCH_SIZES):

    num_patches_per_side = (
        IMAGE_SIZE // patch_size
    )

    plt.subplot(
        1,
        3,
        index + 1
    )

    plt.imshow(sample_image)

    for i in range(
        1,
        num_patches_per_side
    ):

        position = (
            i * patch_size
        )

        plt.axhline(
            position,
            linewidth=1
        )

        plt.axvline(
            position,
            linewidth=1
        )

    plt.title(
        f"{patch_size}x{patch_size} patches"
    )

    plt.axis("off")


plt.tight_layout()

plt.savefig(
    "results/patch_visualization.png"
)

plt.show()


# ============================================================
# 12. SAVE SUMMARY
# ============================================================

best_result = results_df.loc[
    results_df["Test Accuracy"].idxmax()
]

summary = f"""
Vision Transformer Patch Size Comparison
=========================================

Dataset:
CIFAR-10

Patch Sizes:
4x4, 8x8, 16x16

Best Patch Size:
{best_result["Patch Size"]}

Best Test Accuracy:
{best_result["Test Accuracy"]:.4f}

Interpretation:
Smaller patches produce more image tokens.
This allows the Vision Transformer to capture
finer spatial details but increases computation.

Larger patches produce fewer tokens.
This reduces computation but may lose some
fine-grained spatial information.
"""

with open(
    "results/summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(summary)


print("\n")
print("=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)

print(
    "Results saved inside the 'results' folder."
)