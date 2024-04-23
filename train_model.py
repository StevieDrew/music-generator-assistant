import argparse
import os
import random
import time

import mido
import pretty_midi

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



def get_args():
    parser = argparse.ArgumentParser(description="Train an LSTM model.")
    parser.add_argument('--traindatapath', type=str, default='data/classical-midi/all', help='Path to data used to train model.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model.')
    parser.add_argument('--modelname', type=str, default='Model1', help='Name for trained model.')
    return parser.parse_args()



def load_midi_data(midi_path, fs=100):
    """Load MIDI data and return PrettyMIDI object, handling potential errors."""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        return midi_data
    except Exception as e:  # Catching a generic exception if the specific one is unknown
        print(f"An error occurred while loading {midi_path}: {e}")
        return None



"""
Calculate number of frames per bar based on tempo and time signature.

def get_frames_per_bar(tempo, time_signature, fs):
    beats_per_bar = time_signature.numerator
    beat_length = 60 / tempo  # seconds per beat
    frames_per_beat = fs * beat_length
    frames_per_bar = frames_per_beat * beats_per_bar
    return int(frames_per_bar)
"""



"""
def process_multiple_midi_files(midi_paths, fs=100, fixed_length=128):
    input_sequences = []
    output_sequences = []
    for midi_path in midi_paths:
        start_time = time.time()  # Start time for processing this file
        midi_data = load_midi_data(midi_path, fs)
        if midi_data is not None:
            piano_roll = midi_data.get_piano_roll(fs=fs).T
            for i in range(len(piano_roll) - fixed_length):
                input_sequences.append(piano_roll[i:i+fixed_length])
                output_sequences.append(piano_roll[i+1:i+fixed_length+1])
            elapsed_time = time.time() - start_time  # Time taken to process the file
            print(f"Completed processing {midi_path} in {elapsed_time:.2f} seconds")
        else:
            print(f"Failed to load or process {midi_path}")

    input_sequences = np.array(input_sequences, dtype=np.float32)
    output_sequences = np.array(output_sequences, dtype=np.float32)

    print(f"Final shape of all input sequences: {input_sequences.shape}")
    print(f"Final shape of all output sequences: {output_sequences.shape}")

    return input_sequences, output_sequences
"""


def data_generator(midi_paths, fs=100, fixed_length=128, batch_size=32):
    while True:  # Loop forever so the generator never terminates
        inputs, targets = [], []
        for midi_path in midi_paths:
            midi_data = load_midi_data(midi_path, fs)
            if midi_data is not None:
                piano_roll = midi_data.get_piano_roll(fs=fs).T / 127
                for i in range(0, len(piano_roll) - fixed_length, fixed_length):  # non-overlapping segments
                    inputs.append(piano_roll[i:i+fixed_length])
                    targets.append(piano_roll[i+1:i+fixed_length+1])
                    if len(inputs) == batch_size:
                        yield (np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32))
                        inputs, targets = [], []  # Reset for next batch
        if inputs:  # Yield any remaining data as the last batch
            yield (np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32))



def count_samples(midi_files, fs=100, fixed_length=128):
    total_samples = 0
    for midi_path in midi_files:
        midi_data = load_midi_data(midi_path, fs)
        if midi_data is not None:
            piano_roll = midi_data.get_piano_roll(fs=fs).T
            total_samples += (piano_roll.shape[0] - fixed_length) // fixed_length
    return total_samples


"""
def split_data(input_sequences, output_sequences, train_ratio=0.8):
    # Calculate the split index based on the specified ratio
    split_index = int(len(input_sequences) * train_ratio)
    print("Splitting data and converting to NumPy arrays.")
    # Split the sequences without shuffling
    train_inputs = input_sequences[:split_index]
    train_outputs = output_sequences[:split_index]
    test_inputs = input_sequences[split_index:]
    test_outputs = output_sequences[split_index:]
    
    return train_inputs, train_outputs, test_inputs, test_outputs
"""



def build_model(epochs, modelname):
    # Initialize model
    model = Sequential([
        LSTM(256, input_shape=(None, 128), return_sequences=True),  # 128 is the number of pitches
        TimeDistributed(Dense(128, activation='sigmoid'))  # Output layer, 128 pitches
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    checkpoint = ModelCheckpoint(
        modelname + '.keras',       # path where the model is saved
        save_best_only=True,        # only save when the 'val_loss' is improved
        monitor='loss',         # metric to monitor
        mode='min'                  # save the model when the monitored metric decreases
    )

    early_stopping = EarlyStopping(
        monitor='loss',         # metric to monitor
        patience=5,                 # number of epochs with no improvement after which training will be stopped
        verbose=1,                  # verbosity mode
        mode='min'                  # training will stop when the quantity monitored has stopped decreasing
    )


    return model, checkpoint, early_stopping




def main(folder_path, epochs, modelname, fs=100, fixed_length=128, train_ratio=0.8, batch_size=32):
    # Load MIDI files from the specified folder
    train_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.mid') or file.endswith('.midi')]

    # Create separate generators for training and testing
    train_generator = data_generator(train_files, fs, fixed_length, batch_size)

    total_train_samples = count_samples(train_files, fs, fixed_length)
    steps_per_epoch = total_train_samples // batch_size

    model, checkpoint, early_stopping = build_model(epochs, modelname)

    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save model to models folder
    model.save('models/' + modelname + '.keras')





   
if __name__ == "__main__":
    args = get_args()
    main(args.datapath, args.epochs, args.modelname)



    # scp ~/Documents/Projects/CapstoneMusicGenerator/train_model.py smora104@diamond.cs.fiu.edu:~/CapstoneProject/myenv