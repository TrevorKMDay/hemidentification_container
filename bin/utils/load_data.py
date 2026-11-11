
import pickle as pkl
import pprint as pp
import re
import sys

import numpy as np
import pandas as pd


def check_regex_in_cx(cx, regex):
    '''
    Instead of trying to do complicated checks of whether the regex is bounded
    by '_' or the end of the string, just break the standard format up by
    underscore and then check the two ROI names individually. 
    '''

    # Split cx 
    _, x, y = cx.split("_")

    # If the string is detected in either ROI, return True
    result = re.search(re.compile(regex), x) or re.search(re.compile(regex), y)
    return(result)

def load_data(data_to_load, keep_pairs=None, cols_to_keep=None, 
              cols_regex=None):

    if ".pickle" in data_to_load:

        print(f"INFO: Loading data from file: {data_to_load}")
        with open(data_to_load, "rb") as f:
            data = pkl.load(f)

    elif ".csv" in data_to_load:
        data = pd.read_csv(data_to_load)

    else:

        # TO DO: Add CSVs
        print(f"ERROR: Data format of {data_to_load} not recognized, exiting")
        sys.exit(1)

    print(f"INFO: The data has dimensions: {data.shape}")

    # Fix this to display column names not NaN count
    cx = data.filter(regex = r'^cx_', axis=1)

    # Get the columns to keep - do this before NaN replacement
    if cols_to_keep is not None:
        cols_in_cx = [x in cx.columns for x in cols_to_keep]
        if not all(cols_in_cx):
            print("ERROR: The following columns are not in the data, aborting.")
            for i in cols_to_keep:
                if i not in cx.columns:
                    print(f"  Missing: {i}")
            return(None)

    if cols_regex is not None:
        # print(cols_regex) 
        # sys.exit()
        cols_to_keep = [s for s in cx.columns 
                        if check_regex_in_cx(s, cols_regex)]
        if len(cols_to_keep) == 0:
            print(f"ERROR: Regex ({cols_regex}) found no columns, aborting")
            sys.exit(1)
        else:
            print(f"INFO: Found {len(cols_to_keep)} columns based on regex, " + 
                  "continuing")

    if cols_to_keep is not None:
        pp.pprint(cols_to_keep)
        cx = cx[cols_to_keep]
        print("INFO: After filtering, the connection matrix has " + 
              f"dimensions: {cx.shape}")

    nans_in_input = cx.isna().sum()
    nans_in_input_names = [col for x, col in zip(nans_in_input, cx.columns)
                           if x > 0]

    if len(nans_in_input_names) > 0:
        print(f"\nWARNING: {len(nans_in_input_names)} connection columns have "
              "at least 1 NaN value")

    if len(nans_in_input_names) > 0:

        all_nans = [col for x, col in zip(nans_in_input, cx.columns)
                    if x > cx.shape[0] / 2]
        lt50pct_nans = [(x, col) for x, col in zip(nans_in_input, cx.columns)
                        if x <= cx.shape[0] / 2 and x > 0]

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

    # Subset data
    if keep_pairs is not None:

        keep_matrix = pd.DataFrame(data = None,
                                    index=range(data.shape[0]),
                                    columns=range(len(keep_pairs)))

        for i, (col, value) in enumerate(keep_pairs):

            if col not in data.columns:
                print(f"WARNING: Did not find group column '{col}' in data, ",
                        "skipping this column.")
                continue

            # print(value)
            if value.startswith("!"):
                value = re.sub("^!", "", value)
                print(f" -> Inverting relation and keeping rows where {col}",
                      f"is not '{value}'")
                i_rows_to_keep = [x != value for x in data[col]]
                op = "!="
            else:
                i_rows_to_keep = [x == value for x in data[col]]
                op = "="

            n_removed = len(i_rows_to_keep) - sum(i_rows_to_keep)
            print(f" -> Removed {n_removed} rows based on {col} {op} '{value}'")

            keep_matrix.iloc[:, i] = i_rows_to_keep

        # Does this row meet all conditions?
        rows_to_keep = keep_matrix.all(axis=1)

        data = data[rows_to_keep]
        cx = cx[rows_to_keep]

    else:
        print("INFO: No subsetting requested")

    id_cols = data.drop(cx.columns, axis=1)
    n_id = id_cols.shape[1]
    print(f" -> I found {n_id} ID columns: {id_cols.columns.tolist()}")

    assert id_cols.shape[0] > 0, "ERROR: No rows returned after filtering " + \
        "ID columns, check filtering."

    assert cx.shape[0] > 0, "ERROR: No rows returned after filtering " + \
        "connection columns, check filtering."


    return((id_cols, cx))