import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        self.W_hh = np.eye(hidden_size)
        self.W_xh = np.array([[1]])
        self.b_h = np.zeros((1, hidden_size))
        self.hidden_size = hidden_size

    def forward(self, x):
        hidden_states = []
        h_t = np.zeros((1, self.hidden_size))

        for i in range(len(x)):
            h_t = np.tanh(np.dot(h_t, self.W_hh) + np.dot(x[i], self.W_xh) + self.b_h)
            hidden_states.append(h_t.flatten())

        return hidden_states

"""

----------------- Synthetic Data ------------------------

"""
# Generate input data
input_data = np.array([[-1], [0], [1], [2], [3], [4]])

# Create and train the SimpleRNN model
model = SimpleRNN(input_size=1, hidden_size=1)
hidden_states = model.forward(input_data)

# Print the hidden states
for i, hidden_state in enumerate(hidden_states):
    print(f'Timestep {i}: {hidden_state}')

# Plot the hidden states
plt.figure()
for i, hidden_state in enumerate(hidden_states):
    plt.scatter(np.full_like(hidden_state, i), hidden_state, label=f'Timestep {i}')

plt.xlabel('Timestep')
plt.ylabel('Hidden State Value')
plt.ylim([-1.1, 1.1])  # Adjust the y-axis limits
plt.legend()
plt.show()



"""
------------------ Times Series DataSet -----------------------
"""


# Load data from CSV file
file_path = 'Microsoft_Stock.csv'
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M:%S')

# Plot time series
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue')
plt.title('Microsoft Stock Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()
