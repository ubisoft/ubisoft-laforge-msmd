import numpy as np
import pickle as pkl
import os
import lmdb
from tqdm import tqdm

def save_dict_in_chunks(data, file_path, chunk_size=100):
    """
    Save a dictionary in chunks to a pickle file.

    Parameters:
    - data: The large dictionary to save.
    - file_path: Path where to save the pickle file.
    - chunk_size: Number of key-value pairs to include in each chunk.
    """
    with open(file_path, 'wb') as f:
        keys = list(data.keys())
        for i in tqdm(range(0, len(keys), chunk_size)):
            chunk = {k: data[k] for k in keys[i:i + chunk_size]}
            pkl.dump(chunk, f)

def load_dataset_pkl(file_path):
    def load_dict_in_chunks(file_path):
        """
        Load a dictionary in chunks from a pickle file.
        """
        with open(file_path, 'rb') as f:
            while True:
                try:
                    chunk = pkl.load(f)
                    yield chunk
                except EOFError:
                    break  # End of file reached

    # Example usage:
    loaded_dict = {}
    for chunk in load_dict_in_chunks(file_path):
        loaded_dict.update(chunk)  # Combine chunks into a single dictionary
    return loaded_dict


if __name__ == "__main__":
    DATASET_ROOT = "/data/celebv-text/"
    PROCESSED_DATA_ROOT = os.path.join(DATASET_ROOT, "processed_data")
    
    input_lmdb_path = os.path.join(PROCESSED_DATA_ROOT, "processed_data_30fps_v3.lmdb")
    output_pickle_path = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_toy_v3.pkl".format(30))
    output_lmdb_path = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_toy_v3.lmdb".format(30))
    output_key_list = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_toy_v3_keys.txt".format(30))
    output_pickle_path_full = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_v3.pkl".format(30))
    output_lmdb_path_full = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_v3.lmdb".format(30))
    output_key_list_full = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_v3_keys.txt".format(30))

    output_key_list_train = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_toy_v3_keys_train.txt".format(30))
    output_key_list_valid = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_toy_v3_keys_valid.txt".format(30))
    output_key_list_test = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_toy_v3_keys_test.txt".format(30))

    output_key_list_train_full = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_v3_keys_train.txt".format(30))
    output_key_list_valid_full = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_v3_keys_valid.txt".format(30))
    output_key_list_test_full = os.path.join(PROCESSED_DATA_ROOT, "processed_data_{}fps_v3_keys_test.txt".format(30))


    error_files_path = os.path.join(PROCESSED_DATA_ROOT, "error_files.pkl")
    valid_file_ids_path = os.path.join(DATASET_ROOT, "keys.txt")

    pickle_30fps = {}
    error_files = []    

    all_valid_files = []
    with open(valid_file_ids_path, "r") as f:
        all_valid_files = f.read().splitlines()
    
    # randomly order all_valid_files
    np.random.seed(42)
    np.random.shuffle(all_valid_files)
    
    print(len(all_valid_files))
    env = lmdb.open(input_lmdb_path, map_size=1099511627776)
    # get the size of the lmdb
    print("Size of the lmdb: ", env.stat()["entries"])
    ############################################################################################################
    ############################################# building toy set #############################################
    ############################################################################################################
    save_toy_set = True
    if save_toy_set:
        env = lmdb.open(input_lmdb_path, map_size=1099511627776)
        txn = env.begin()
        # take out 1000 files as the debugging set
        toy_set_dict = {}
        toy_set_keys = []
        count = 1000
        counter = 0
        i = 0
        for i in tqdm(range(len(all_valid_files))):
            if counter == count:
                break
            key = all_valid_files[i]
            # try and search in lmdb
            value = txn.get(key.encode())
            if not value is None:
                counter += 1
                toy_set_dict[key] = pkl.loads(value)
                toy_set_keys.append(key)
        env.close()

        # write the toy set to lmdb
        print(output_lmdb_path)
        env = lmdb.open(output_lmdb_path, map_size=1099511627776)
        txn = env.begin(write=True)
        for key in tqdm(toy_set_keys):
            txn.put(key.encode(), pkl.dumps(toy_set_dict[key]))
        txn.commit()
        env.close()

        # write the toy set to pickle
        with open(output_pickle_path, "wb") as f:
            pkl.dump(toy_set_dict, f)
        
        # write the keys to a file
        with open(output_key_list, "w") as f:
            f.write("\n".join(toy_set_keys))
        
        # do a split
        train_split = 0.8
        val_split = 0.1
        test_split = 0.1
        np.random.seed(42)
        np.random.shuffle(toy_set_keys)
        train_keys = toy_set_keys[:int(len(toy_set_keys) * train_split)]
        val_keys = toy_set_keys[int(len(toy_set_keys) * train_split):int(len(toy_set_keys) * (train_split + val_split))]
        test_keys = toy_set_keys[int(len(toy_set_keys) * (train_split + val_split)):]

        # write the keys to separate files  
        with open(output_key_list_train, "w") as f:
            f.write("\n".join(train_keys))
        with open(output_key_list_valid, "w") as f:
            f.write("\n".join(val_keys))
        with open(output_key_list_test, "w") as f:
            f.write("\n".join(test_keys))
        import time
        # time the loading of the lmdb toy set
        start = time.time()
        env = lmdb.open(output_lmdb_path, map_size=1099511627776)
        txn = env.begin()
        for key in tqdm(toy_set_keys):
            value = txn.get(key.encode())
            if value is None:
                print("Error: key not found in lmdb")
                error_files.append(key)
        env.close()
        end = time.time()
        print("Time taken to load the toy set: ", end - start)

        start = time.time()
        with open(output_pickle_path, "rb") as f:
            toy_set_dict = pkl.load(f)
        end = time.time()
        print("Time taken to load the toy set from pickle: ", end - start)
        

    ############################################################################################################
    ############################################# building real set (v2) ############################################
    ############################################################################################################
    

    dict_to_save = {}
    error_files = []
    env = lmdb.open(input_lmdb_path, map_size=1099511627776) # 1TB
    txn = env.begin()
    for key in tqdm(all_valid_files):
        value = txn.get(key.encode())
        if not value is None:
            dict_to_save[key] = pkl.loads(value)
        else:
            error_files.append(key)
    env.close()    
    save_dict_in_chunks(dict_to_save, output_pickle_path_full, chunk_size=100)
    
    # do train test split
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1
    np.random.seed(42)
    np.random.shuffle(all_valid_files)
    all_valid_files = list(dict_to_save.keys())
    np.random.shuffle(all_valid_files)
    train_keys = all_valid_files[:int(len(all_valid_files) * train_split)]
    val_keys = all_valid_files[int(len(all_valid_files) * train_split):int(len(all_valid_files) * (train_split + val_split))]
    test_keys = all_valid_files[int(len(all_valid_files) * (train_split + val_split)):]

    # save the files
    with open(output_key_list_full, "w") as f:
        f.write("\n".join(all_valid_files))
    with open(output_key_list_train_full, "w") as f:
        f.write("\n".join(train_keys))
    with open(output_key_list_valid_full, "w") as f:
        f.write("\n".join(val_keys))
    with open(output_key_list_test_full, "w") as f:
        f.write("\n".join(test_keys))

    # test overlap between sets
    train_set = set(train_keys)
    val_set = set(val_keys)
    test_set = set(test_keys)
    print("Overlap between train and val: ", len(train_set.intersection(val_set)))
    print("Overlap between train and test: ", len(train_set.intersection(test_set)))
    print("Overlap between val and test: ", len(val_set.intersection(test_set)))
