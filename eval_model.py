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
    parser.add_argument('--testdatapath', type=str, default='data/classical-midi/test', help='Path to data used to train model.')
    parser.add_argument('--modelname', type=str, default='Model1', help='Name for trained model.')
    return parser.parse_args()




def load_midi_data(midi_path, fs):
    """ Attempts to load MIDI data and handle errors gracefully. """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        return midi_data
    except Exception as e:
        print(f"An error occurred while loading {midi_path}: {e}")
        return None




# Load the model
def load_keras_model(modelname):
    model = tf.keras.models.load_model(modelname)
    return model




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




def get_midi_length(midi_path, fs=100):
    """ Returns the length of a MIDI file in terms of the number of time steps in its piano roll, or None if loading fails. """
    midi_data = load_midi_data(midi_path, fs)
    if midi_data is None:
        return None
    piano_roll = midi_data.get_piano_roll(fs=fs)
    return piano_roll.shape[1]




def calculate_sequences_per_file(midi_length, fixed_length=128, step_size=128):
    """ Calculate how many non-overlapping sequences can be generated from a MIDI file of a given length. """
    if midi_length is None or midi_length < fixed_length:
        return 0  # Return 0 sequences if the MIDI length is invalid or too short
    return (midi_length - fixed_length) // step_size + 1




def average_sequences(midi_files, fs=100, fixed_length=128, step_size=128):
    total_sequences = 0
    count = 0
    for midi_path in midi_files:
        midi_length = get_midi_length(midi_path, fs)
        sequences = calculate_sequences_per_file(midi_length, fixed_length, step_size)
        if sequences > 0:
            total_sequences += sequences
            count += 1
    return total_sequences / count if count else 0







def main(folder_path, modelname):
    # Load MIDI files from the specified folder
    test_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.mid') or file.endswith('.midi')]
    
    random.seed(42)

    average_sequences_per_file = average_sequences(test_files)
    print(f"Average sequences per file: {average_sequences_per_file}")
    number_of_test_files = len(test_files)  # Assume this is 200

    total_test_sequences = average_sequences_per_file * number_of_test_files
    batch_size = 32  # Your chosen batch size

    steps_per_epoch_test = total_test_sequences // batch_size
    if total_test_sequences % batch_size != 0:
        steps_per_epoch_test += 1  # Make sure to cover all samples


    test_generator = data_generator(test_files, fs=100, fixed_length=128, batch_size=32)

    model = load_keras_model(modelname)

    results = model.evaluate(test_generator, steps=steps_per_epoch_test)  
    
    print("Test Loss, Test Accuracy:", results)

    



if __name__ == "__main__":
    args = get_args()
    main(args.testdatapath, args.modelname)