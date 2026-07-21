
import pickle as pkl
import numpy as np
import pprint as pp

def load_data(data_to_load, col=None, col_values=[]):

    # Load data
    if ".pickle" in data_to_load:

        with open(data_to_load, "rb") as f:
            data = pkl.load(f)

    else:

        # TO DO: Add CSVs
        print(f"Data format of {data_to_load} not recognized, exiting")
        exit(1)

    print(f"The data has dimensions: {data.shape}")

    if col is not None and col not in data.columns:
        print(f"WARNING: Did not find group column '{col}' in data, not",
              "subsetting.")

    ## Extract cx columns

    cx = data.filter(regex='^cx_', axis=1)
    n_cx = cx.shape[1]
    print(f" -> I found {n_cx} connections.")

    ## Check cx dimensionality
    corr_mat_size = 0.5 * (np.sqrt(8 * n_cx + 1) + 1)

    if not corr_mat_size.is_integer():
        # TO DO: Add option to skip this
        print("The input is expected to be one-half of a correlation matrix.")
        print(f"This does not seem to be (size: {corr_mat_size}), exiting.")
        # print("Use --no-check-size to skip this check")

    if col in data.columns:
        rows_to_keep = data[col].isin(col_values)
        print(f" -> Kept {sum(rows_to_keep)}/{cx.shape[0]} training rows.")

        data = data[rows_to_keep]
        cx = cx[rows_to_keep]

    id_cols = data.drop(cx.columns, axis=1)
    n_id = id_cols.shape[1]

    print(f" -> I found {n_id} ID columns:")
    pp.pprint(id_cols.columns.tolist(), compact=True)

    return((id_cols, cx))