from tensorflow.keras import layers, models, regularizers, optimizers, callbacks


def build_model(num_classes: int, input_shape=(128, 128, 3)):
    model = models.Sequential([
        layers.RandomFlip("horizontal", input_shape=input_shape),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),

        layers.Conv2D(32, 3, activation="relu",
                       kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu",
                       kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(checkpoint_path: str):
    return [
        callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
    ]