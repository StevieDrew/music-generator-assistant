import os
import shutil
import random
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Split your data to train and test. Run this before training your model.")
    parser.add_argument('--datapath', type=str, default='data/classical-midi/all', help='Path to data used to train and test model.')
    return parser.parse_args()

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def copy_files(file_list, target_directory):
    for file in file_list:
        shutil.copy(file, os.path.join(target_directory, os.path.basename(file)))

def main(folder_path):
    midi_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.mid') or file.endswith('.midi')]
    
    random.seed(42)
    random.shuffle(midi_files)

    split_index = int(len(midi_files) * 0.8)  # 80% for training, 20% for testing
    train_files = midi_files[:split_index]
    test_files = midi_files[split_index:]

    # Ensure directories
    ensure_directory_exists('data/classical-midi/train')
    ensure_directory_exists('data/classical-midi/test')

    # Move files
    copy_files(train_files, 'data/classical-midi/train')
    copy_files(test_files, 'data/classical-midi/test')

if __name__ == "__main__":
    args = get_args()
    main(args.datapath)
