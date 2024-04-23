import argparse
import pretty_midi
import numpy as np
import tensorflow as tf

def get_args():
    parser = argparse.ArgumentParser(description="Generate output from a trained LSTM model.")
    parser.add_argument('--midifile', type=str, required=True, help='Path to the MIDI file for generating output.')
    parser.add_argument('--modelpath', type=str, required=True, help='Path to the trained model.')
    return parser.parse_args()

def load_midi_data(midi_path, fs=100):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        piano_roll = midi_data.get_piano_roll(fs=fs).T / 127
        return piano_roll
    except Exception as e:
        print(f"Error loading MIDI file {midi_path}: {e}")
        return None

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def generate_output(model, input_data, fixed_length=128):
    # Start with initial input
    current_input = input_data[:fixed_length]
    output_sequence = np.copy(current_input)

    # Model expects input data to be normalized or preprocessed as during training
    current_input = np.expand_dims(current_input, axis=0)  # Add batch dimension

    # Generate sequence double the length of the input
    while len(output_sequence) < 2 * fixed_length:
        prediction = model.predict(current_input)
        # Append the prediction to output_sequence
        predicted_next_step = prediction[:, -1, :]  # Assuming the model predicts step-by-step
        output_sequence = np.concatenate((output_sequence, predicted_next_step), axis=0)

        # Update current_input to include newly predicted part
        current_input = np.expand_dims(output_sequence[-fixed_length:], axis=0)  # Slide window

    return output_sequence[:2 * fixed_length]  # Ensure the output is exactly twice the input length


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    """ Convert a Piano Roll array to a PrettyMIDI object.
        :param piano_roll: np.ndarray, shape=(num_pitches, num_frames), dtype=bool
        :param fs: int, frames per second
        :param program: The MIDI program (instrument) number (default=0 for Acoustic Grand Piano).
        :return: A pretty_midi.PrettyMIDI class instance with the converted data.
    """
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # Pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # Use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll.astype(int), axis=1))
    prev_velocities = np.zeros(piano_roll.shape[1])
    note_times = np.zeros(piano_roll.shape[0])

    for time, note in zip(*velocity_changes):
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_times[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=int(prev_velocities[note]),
                pitch=note,
                start=note_times[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm



def save_midi(path, midi_data):
    """ Save a PrettyMIDI object to a MIDI file.
        :param path: str, path to save the MIDI file
        :param midi_data: pretty_midi.PrettyMIDI, the MIDI data to save
    """
    midi_data.write(path)




def main(midifile, modelpath):
    midi_data = load_midi_data(midifile)
    if midi_data is not None:
        model = load_model(modelpath)
        output = generate_output(model, midi_data)
        print("Generated Output:", output)

        # Convert output back to MIDI
        output_midi = piano_roll_to_pretty_midi(output, fs=100)
        
        # Save the MIDI file
        save_midi('output.mid', output_midi)
        print("Output MIDI saved as 'output.mid'.")



if __name__ == "__main__":
    args = get_args()
    main(args.midifile, args.modelpath)
