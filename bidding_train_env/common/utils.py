import os
import pickle


def save_normalize_dict(normalize_dict, save_dir):
    """Save the normalization dictionary to a Pickle file."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'normalize_dict.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(normalize_dict, file)
