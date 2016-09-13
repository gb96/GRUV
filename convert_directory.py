from data_utils import parse_files
import config.nn_config as nn_config


config = nn_config.get_neural_net_configuration()
input_directory = config['dataset_directory']
output_filename = config['model_file']

freq = config['sampling_frequency']  # sample frequency in Hz
clip_len = 10  # length of clips for training. Defined in seconds
# block sizes used for training - this defines the size of our input state
block_size = freq / 4
# Used later for zero-padding song sequences
max_seq_len = int(round((freq * clip_len) / block_size))
# Step 1 - convert MP3s to WAVs
#
new_directory = parse_files.convert_folder_to_wav(input_directory, freq)
# Step 2 - convert WAVs to frequency domain
# with mean 0 and standard deviation of 1
parse_files.wav_files_to_nptensor(new_directory, block_size,
                                  max_seq_len, output_filename)
