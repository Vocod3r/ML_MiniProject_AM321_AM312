"""
The four CNN versions from the paper (Figures 2 and 3).

The paper doesn't give filter counts or kernel sizes, so we use
3x3 kernels, 32 then 64 filters, and 'same' padding. Report these choices.

  V1: conv32 -> pool -> conv64 -> pool -> flatten -> dense128 -> softmax   (full model)
  V2: conv32 -> pool -> flatten -> dense128 -> softmax                     (2nd conv + its pool removed)
  V3: conv32 -> flatten -> dense128 -> softmax                             (remaining pool removed)
  V4: conv32 -> flatten -> dense64  -> softmax                             (dense 128 -> 64)

Note: Figure 3 is ambiguous about whether V3/V4 build on V2 or on V1.
We read it as progressive simplification (each version builds on the last).
Removing the pool makes the flattened layer 4x bigger, so V3 may have MORE
parameters than V2 -- worth discussing in the report.
"""
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(version, n_classes, filters=(32, 64), kernel=3):
    m = keras.Sequential(name=f"cnn_{version}")
    m.add(keras.Input(shape=(28, 28, 1)))
    m.add(layers.Conv2D(filters[0], kernel, padding="same", activation="relu"))

    if version == "V1":
        m.add(layers.MaxPooling2D(2))
        m.add(layers.Conv2D(filters[1], kernel, padding="same", activation="relu"))
        m.add(layers.MaxPooling2D(2))
        dense_units = 128
    elif version == "V2":
        m.add(layers.MaxPooling2D(2))
        dense_units = 128
    elif version == "V3":
        dense_units = 128
    elif version == "V4":
        dense_units = 64
    else:
        raise ValueError(f"Unknown version {version}; use V1, V2, V3 or V4")

    m.add(layers.Flatten())
    m.add(layers.Dense(dense_units, activation="relu"))
    m.add(layers.Dense(n_classes, activation="softmax"))
    return m


if __name__ == "__main__":
    for v in ["V1", "V2", "V3", "V4"]:
        print(v, f"{build_cnn(v, 50).count_params():,} parameters")