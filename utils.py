import h5py

def read_dataset(dataset_path, verbose):
    f = h5py.File(dataset_path, 'r')

    if verbose:
        print("Reading X...")
    X = f['X']
    if verbose:
        print("Reading Y...")
    Y = f['Y']

    if verbose:
        print(X.shape, Y.shape)

    return X, Y, f

    # Not close the file here
    # f.close()