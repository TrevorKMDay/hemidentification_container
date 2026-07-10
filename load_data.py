import pickle as pkl
import numpy as np

def load_data(data_to_load):

    if ".pickle" in data_to_load:

        with open(data_to_load, "rb") as f:
            data = pkl.load(f)

    else:

        # TO DO: Add CSVs
        print(f"Data format of {data_to_load} not recognized, exiting")
        exit(1)

    print(f"The data has dimensions: {data.shape}")

    ## Extract cx columns

    cx = data.filter(regex='^cx_', axis=1)
    n_cx = cx.shape[1]
    print(f" -> I found {n_cx} connections.")

    ## Check cx dimensionality
    corr_mat_size = 0.5 * (np.sqrt(8 * n_cx + 1) + 1)

    if not corr_mat_size.is_integer():
        print("The input is expected to be one-half of a correlation matrix.")
        print(f"This does not seem to be, size: {corr_mat_size}, exiting.")
        print("Use --no-check-size to skip this check")

    id_cols = data.drop(cx.columns, axis=1)
    n_id = id_cols.shape[1]
    print(f" -> I found {n_id} ID columns: {id_cols.columns.values}.")

    return((id_cols, cx))