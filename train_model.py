import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Binary output: Parasitized vs Uninfected
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the model - this creates the file your Flask app is looking for
model.save('maleria_MD.h5')
print("Model saved successfully as 'maleriaMD.h5'")