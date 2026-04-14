import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Create a placeholder LSTM model
# Input shape: (batch_size, 1, 13) - 13 features
model = Sequential([
    LSTM(64, activation='relu', input_shape=(1, 13), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')  # Output AQI value
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Create dummy training data for demonstration
X_dummy = np.random.randn(100, 1, 13)
y_dummy = np.random.randn(100, 1) * 100 + 50  # Random AQI values (0-200 range)

# Train briefly to initialize weights
model.fit(X_dummy, y_dummy, epochs=5, batch_size=16, verbose=0)

# Save the model
model.save('lstm_model.h5')
print("✓ LSTM model created and saved as lstm_model.h5")
