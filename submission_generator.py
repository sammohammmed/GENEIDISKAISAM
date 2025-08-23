import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

print("Stanford RNA 3D Folding Competition - Submission Generator (GNN Model)")

# Load test sequences from Kaggle or use fallbacks
try:
    input_path = Path('/kaggle/input/stanford-rna-3d-folding')
    test_files = list(input_path.glob('*test*.csv'))
    if test_files:
        test_df = pd.read_csv(test_files[0])
        print(f"Loaded {len(test_df)} test sequences from competition data")
        test_sequences = [(row['ID'], row['sequence']) for _, row in test_df.iterrows()]
    else:
        raise FileNotFoundError("No test files found in the expected path.")
except Exception as e:
    print(f"Could not load competition data: {e}")
    # Fallback sequences for demonstration
    test_sequences = [
        ('R1107', 'GGGGGCCACAGCAGAAGCGUUCACGUCGCAGCCCCUGUCAGCCAUUGCACUCCGGCUGCGAAUUCUGCU'),
        ('R1108', 'GGGGGCCACAGCAGAAGCGUUCACGUCGGCCCCUGUCAGCCAUUGCACUCCGGCUGCGAAUUCUGCU')
    ]
    print(f"Using {len(test_sequences)} fallback sequences")

# --- Model Definition (Simplified Graph Neural Network) ---

class SimpleGNN(nn.Module):
    """
    A simplified GNN model to predict 3D coordinates from a sequence.
    This is a conceptual model and requires training to be effective.
    """
    def __init__(self, num_residues, embed_dim, hidden_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.embedding = nn.Embedding(num_residues, embed_dim)

        # A simple GNN-like layer using a fully connected network
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is a tensor of residue indices
        embedded = self.embedding(x)

        # Simple message passing-like step (conceptual)
        x = F.relu(self.fc1(embedded))

        # Final prediction for coordinates (x, y, z)
        x = self.fc2(x)
        return x

# Map nucleotides to integer IDs for the embedding layer
nuc_map = {'A': 0, 'U': 1, 'C': 2, 'G': 3}

# Model parameters
embed_dim = 16
hidden_dim = 32
output_dim = 3 # x, y, z coordinates

# Instantiate and "load" a pre-trained model (for demonstration only)
# In a real competition, you would train this model on training data.
dummy_model = SimpleGNN(len(nuc_map), embed_dim, hidden_dim, output_dim)
# We will use random weights to simulate a model for demonstration
# A real solution would load trained weights here.
dummy_model_state = {
    'embedding.weight': torch.randn(len(nuc_map), embed_dim),
    'fc1.weight': torch.randn(hidden_dim, embed_dim),
    'fc1.bias': torch.randn(hidden_dim),
    'fc2.weight': torch.randn(output_dim, hidden_dim),
    'fc2.bias': torch.randn(output_dim),
}
dummy_model.load_state_dict(dummy_model_state)
dummy_model.eval() # Set to evaluation mode

# --- Submission Generation with the GNN Model ---

submission_rows = []

for seq_id, sequence in test_sequences:
    seq_len = len(sequence)
    print(f"Processing {seq_id}: {seq_len} residues")

    # Convert sequence to a tensor of numerical IDs
    seq_tensor = torch.tensor([nuc_map[nuc] for nuc in sequence])

    # Generate 5 conformations by adding random noise to model predictions
    row_dict = {'ID': seq_id}
    for conf in range(1, 6):
        # Get base predictions from the model
        with torch.no_grad():
            base_coords = dummy_model(seq_tensor).numpy()

        # Add a small amount of random noise for each conformation
        noise = np.random.normal(0, 0.5, size=base_coords.shape)
        final_coords = base_coords + noise

        # Separate coordinates and round to 3 decimal places
        x_list = np.round(final_coords[:, 0], 3).tolist()
        y_list = np.round(final_coords[:, 1], 3).tolist()
        z_list = np.round(final_coords[:, 2], 3).tolist()

        row_dict[f'x_{conf}'] = x_list
        row_dict[f'y_{conf}'] = y_list
        row_dict[f'z_{conf}'] = z_list

    submission_rows.append(row_dict)

# Convert to final DataFrame for submission
final_rows = []
for row in submission_rows:
    seq_len = len(row['x_1'])
    sequence_from_id = [s for id, s in test_sequences if id == row['ID']][0]

    for i in range(seq_len):
        final_row = {
            'ID': row['ID'],
            'resname': sequence_from_id[i],  # Fill in the correct nucleotide
            'resid': i + 1
        }
        for conf in range(1, 6):
            final_row[f'x_{conf}'] = row[f'x_{conf}'][i]
            final_row[f'y_{conf}'] = row[f'y_{conf}'][i]
            final_row[f'z_{conf}'] = row[f'z_{conf}'][i]
        final_rows.append(final_row)

submission_df = pd.DataFrame(final_rows)
submission_df.to_csv('submission.csv', index=False)
print(f"SUCCESS: Created submission.csv with {len(submission_df)} rows")
print("File saved to submission.csv")
