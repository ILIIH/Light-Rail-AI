import os
import math
import tensorflow as tf

# Constants
CHECKPOINT_DIR = "model"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cp-{epoch:04d}.weights.h5")
SAVE_FREQ_MULTIPLIER = 5

# Ensure the checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_model(model):
    """Save the initial weights of the model."""
    model.save_weights(CHECKPOINT_PATH.format(epoch=0))
    """Save model architecture to json file."""
    model_json = model.to_json()
    with open('my_model.json', 'w') as json_file:
        json_file.write(model_json)

def get_weight_callback(array_size, batch_size):
    """Create a ModelCheckpoint callback to save model weights at regular intervals."""
    n_batches = math.ceil(array_size / batch_size)
    save_freq = SAVE_FREQ_MULTIPLIER * n_batches
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH, 
        verbose=1, 
        save_weights_only=True,
        save_freq=save_freq
    )
    
    return cp_callback
