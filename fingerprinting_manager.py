import argparse as ap
import pickle as pkl
import os
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from load_data import load_data

print("Started")

# Set up argument parsers =======

parser = ap.ArgumentParser()

subparsers = parser.add_subparsers(dest="subcommand")

# Training

train_parser = subparsers.add_parser('train')

train_parser.add_argument("training_data")

train_parser.add_argument("output")

train_parser.add_argument("column", nargs="+",
                          help="Column(s) to predict")

train_parser.add_argument("--no-skip-size", action="store_true",
                          help="Do not verify that the input data came from"
                               "one half of a correlation matrix.")

train_parser.add_argument("--join", "-j", default=".",
                          help="Character to join outcomes on, default: .")

train_parser.add_argument("--keep_groups", "-k", nargs="+", 
                          metavar=("COL", "VALUE"),
                          help="Values from the 'fold' column to keep. The " 
                                "first value is the column ID.")

# Testing

test_parser = subparsers.add_parser('test')

test_parser.add_argument("model",
                         help="Model to use, from this program.")

test_parser.add_argument("test_data")

test_parser.add_argument("output_name")

test_parser.add_argument("--keep_groups", "-k", nargs="+", 
                          metavar=("COL", "VALUE"),
                          help="Values from the 'fold' column to keep. The " 
                                "first value is the column ID.")

args = parser.parse_args()
sc = args.subcommand

if sc is None: 
    parser.print_help()
    exit(1)

print(args)
# exit()

if len(args.keep_groups) >= 2:
    keep_col = args.keep_groups[0]
    keep_values = args.keep_groups[1:]
else:
    print("Must supply at least two arguments to '--keep_groups', received: ",
          f"{args.keep_groups}")
    exit(1)

# Load data ========

if sc == "train":

    j = args.join
    output_name = args.output

    if os.path.exists(output_name):
        print(f"\nERROR: File '{output_name}' already exists! Delete to "
                "continue.")
        exit(1)

    # Load the data as ID and classifier columns
    data_to_load = args.training_data
    id, cx = load_data(data_to_load, col=keep_col, col_values=keep_values)

    # Check columns are in data
    pred_cols = args.column
    for col in pred_cols:
        if col not in id.columns:
            print(f"ERROR: Did not find column {col} in the data, exiting")

    print("\nSetting up prediction")
    # If there is only one column to predict, pass it on
    if len(pred_cols) == 1:
        outcome_name = pred_cols[0]
    else:
        # Otherwise, join the values with an arbitrary column
        print("  -> More than one column to predict")
        outcome_name = j.join(pred_cols)
        outcome_df = id[pred_cols]
        id[outcome_name] = outcome_df.apply(j.join, axis=1)

    print(f"  Predicting groups from {outcome_name}: {set(id[outcome_name])}")

    # Start up the classifier
    clf = LinearDiscriminantAnalysis()
    y = id[outcome_name].tolist()

    clf.fit(cx, y)
    # print(clf.explained_variance_ratio_)

    ex_var = [float(round(x, 3)) for x in clf.explained_variance_ratio_]
    print(f"Explained variance: {ex_var}")

    # Save the outcome file name
    result = (outcome_name, clf)

    with open(output_name, "wb") as f:
        pkl.dump(result, f)

    print(f"\nClassifier saved to {output_name}!")

if sc == "test":

    with open(args.model, "rb") as f:
        outcome_name, clf = pkl.load(f)

    outcome_cols = outcome_name.split('.')

    print(f"Predicting column(s) '{outcome_cols}' from the data.")

    # Load the data as ID and classifier columns
    data_to_load = args.test_data
    id, cx = load_data(data_to_load, col=keep_col, col_values=keep_values)

    # Check that all the columns appear in the data
    if not set(outcome_cols).issubset(id.columns):
        print(f"ERROR: Not all elements of {outcome_cols} appear in the",
              "test data:")
        print(id.columns)
        exit(1)

    y_hat = pd.DataFrame({f"{outcome_name}_predicted": clf.predict(cx)})
    scores = pd.DataFrame(clf.transform(cx))
    scores.columns = [f"LD{x}" for x in range(1, scores.shape[1] + 1)]

    final = pd.concat([id, y_hat, scores], axis=1)

    final.to_csv(args.output_name, index=False)