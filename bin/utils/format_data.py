
import sys

import numpy as np
import pandas as pd


def pivot_matrix(matrix, label, names=None):

    # Only do this to connections
    # matrix = matrix.filter(regex = r'^cx_', axis=1)

    if names is not None:
        matrix.index = names

    nans_in_input = matrix.isna().sum()
    all_empty = [x == (matrix.shape[0] - 1) for x in nans_in_input]
    which_empty = [i for i, val in enumerate(all_empty) if val]

    # THIS ISN'T WORKING
    if len(which_empty) > 0:
        print(f"WARNING: Found all-NaN columns: {which_empty}")
        if names is not None:
            bad_cols = [names[i] for i in which_empty]
            print(f"    Names: {bad_cols}")
        print("    Dropping them from matrix.")
        matrix.drop(index=bad_cols, columns=bad_cols, inplace=True)
        print(f"    New shape: {matrix.shape}")

    # Remove the upper tri, including the diag (k=0)
    m = matrix.mask(np.triu(np.ones(matrix.shape), k=0).astype(bool))
    m = m.reset_index()

    # Pivot longer
    m_long = pd.melt(m, id_vars="index")
    m_long2 = m_long[[not np.isnan(v) for v in m_long["value"]]]

    m_long2 = m_long2.assign(index_hemi = None, variable_hemi = None,
                             wide_name = None)

    # FIX WHY ALL THESE ARE THROWING ERROS
    m_long2.loc[:, "index_hemi"] = [x[0] for x in m_long2["index"]]
    m_long2.loc[:, "variable_hemi"] = [x[0] for x in m_long2["variable"]]

    m_long2.loc[:, "index"] = [x[1:] for x in m_long2["index"]]
    m_long2.loc[:, "variable"] = [x[1:] for x in m_long2["variable"]]

    m_long2.loc[:, "wide_name"] = [f"cx_{i}_{v}" for i, v in
                                   zip(m_long2["index"], m_long2["variable"])]

    m_long3 = m_long2[m_long2["index_hemi"] == m_long2["variable_hemi"]]
    m_long4 = m_long3[["wide_name", "index_hemi", "value"]]

    # print(m_long4)

    # Pivot back wider
    m_wide = m_long4.pivot(columns="wide_name", index="index_hemi",
                           values="value")

    # print(m_wide)

    m_wide.insert(0, "label", label)
    m_wide.columns.name = None
    m_wide = m_wide.reset_index()

    return(m_wide)

def format_input(files, output_name, file_with_names=None):

    if file_with_names is not None:
        names = pd.read_table(file_with_names, header=None).iloc[:, 0].tolist()
        print(f"INFO: I was given {len(names)} names to use.")
        names = [n.replace('_', '') for n in names]

    print(f"INFO: I was given {len(files)} pconn files to load.")
    matrices = [pd.read_table(f, header=None, names=names) for f in files]

    matrices_shapes = [m.shape for m in matrices]
    if len(set(matrices_shapes)) > 1:
        print("ERROR: Mismatched matrices found!")
        sys.exit(1)

    if matrices_shapes[0][0] != matrices_shapes[0][1]:
        print(f"ERROR: The input matrix isn't square! {matrices_shapes[0]}")
        print(matrices_shapes)
        sys.exit(1)
    else:
        print(f"Input dimensions are: {matrices_shapes[0]}")

    pivoted_matrices = [pivot_matrix(m, label=f, names=names) for m, f in
                        zip(matrices, files)]

    pivoted_matrices_shapes = [m.shape for m in pivoted_matrices]
    print(pivoted_matrices_shapes)
    if len(set(pivoted_matrices_shapes)) > 1:
        print("WARNING: Mismatched final training datasets found!")
        print("         This means uneven numbers of all-NaN columns were "
              "dropped from the input connectivity matrices.")
        # sys.exit(1)
        # Add how-to-handle instructions

    # Concatenate matrices
    final_pivot = pd.concat(pivoted_matrices, axis=0)
    # print(final_pivot)
    final_pivot.reset_index(inplace=True, drop=True)
    # print(final_pivot)
    return(final_pivot)