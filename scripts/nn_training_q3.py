"""
this is converting the r code to python code
"""
import pandas as pd
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# load cleaned data

#made df in preliminary-q3-outdated.rmd
data_path = Path(__file__).parent.parent / "data" / "claims-clean-task3.csv"
df = pd.read_csv(data_path)

texts = df["text_clean"].astype(str).values
labels = df["bclass"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=133233,
    stratify=labels,
)

# text vectorization layer as seen in the example file
vectorizer = layers.TextVectorization(
    standardize=None,
    split="whitespace",
    ngrams=None,
    max_tokens=None,
    output_mode="tf_idf",
)

vectorizer.adapt(X_train)


# model builders
def build_model_1():
    """Shallow model with small hidden layer (baseline)."""
    text_input = keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorizer(text_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(25, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(text_input, output)
    return model


def build_model_2():
    """Single wider hidden layer."""
    text_input = keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorizer(text_input)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(text_input, output)
    return model


def build_model_3():
    """Two hidden layers for deeper representation."""
    text_input = keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorizer(text_input)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(text_input, output)
    return model


def build_model_4():
    """Smaller model with stronger dropout."""
    text_input = keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorizer(text_input)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(text_input, output)
    return model


# train models (converted from r code)

models = {
    "model1": build_model_1(),
    "model2": build_model_2(),
    "model3": build_model_3(),
    "model4": build_model_4(),
}

histories = {}
test_accuracies = {}

for name, model in models.items():
    print(f"\nTraining {name}")
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.3,
        epochs=5,
        verbose=1,
    )

    histories[name] = history

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    test_accuracies[name] = acc
    print(f"{name} test accuracy: {acc:.4f}")


# summary table

summary_df = pd.DataFrame(
    {
        "model": list(test_accuracies.keys()),
        "test_accuracy": list(test_accuracies.values()),
    }
)

print("\nTest accuracy summary:")
print(summary_df.to_string(index=False))

#save models

results_dir = Path(__file__).parent.parent / "results" / "models"
results_dir.mkdir(exist_ok=True)

for name, model in models.items():
    save_path = results_dir / f"{name}.h5" #saving as .keras was not working so changed to .h5
    print(f"Saving {name} to {save_path}")
    model.save(save_path)
