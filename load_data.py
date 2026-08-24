
import pickle as pkl
import pprint as pp
import sys

import numpy as np


def load_data(data_to_load, col=None, col_values=None):

    # Load data
    if col_values is None:
        col_values = []

    if ".pickle" in data_to_load:

        with open(data_to_load, "rb") as f:
            data = pkl.load(f)

    else:

        # TO DO: Add CSVs
        print(f"Data format of {data_to_load} not recognized, exiting")
        sys.exit(1)

    print(f"The data has dimensions: {data.shape}")

    if col is not None and col not in data.columns:
        print(f"WARNING: Did not find group column '{col}' in data, not",
              "subsetting.")

    # Fix this to display column names not NaN count
    nans_in_input = data.isna().sum()
    nans_in_input_names = [col for x, col in zip(nans_in_input, data.columns)
                           if x > 0]
    print(f"WARNING: {len(nans_in_input_names)} columns have at least 1 NaN "
          "value")

    if len(nans_in_input_names) > 0:

        all_nans = [col for x, col in zip(nans_in_input, data.columns)
                    if x > data.shape[0] / 2]
        lt50pct_nans = [(x, col) for x, col in zip(nans_in_input, data.columns)
                        if x <= data.shape[0] / 2 and x > 0]

        # pp.pprint(lt50pct_nans, compact=True)

        if len(all_nans) > 0:
            print(f"  Performed repair: {len(all_nans)} columns with more "
                  "than 50% NaNs were removed.")
            pp.pprint(all_nans, compact=True)
            data.drop(columns=all_nans, inplace=True)
            print(f"  Current data size: {data.shape}")

        if len(lt50pct_nans) > 0:

            for x, c in lt50pct_nans:
                mval = np.mean(data[c])
                print(f"Trying to NaN-fill {c} ({x} missing values), ",
                      f"mean: {mval:.2f}")
                data[c] = data[c].fillna(mval)

            print(f"  Performed repair: {len(lt50pct_nans)} columns with less "
                "than 50% NaNs were mean-filled.")

    ## Extract cx columns

    cx = data.filter(regex='^cx_', axis=1)
    n_cx = cx.shape[1]
    print(f" -> I found {n_cx} connections.")

    # print(cx)

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
    # pp.pprint(id_cols.columns.tolist(), compact=True)

    # print([id_cols.shape, cx.shape])
    return((id_cols, cx))