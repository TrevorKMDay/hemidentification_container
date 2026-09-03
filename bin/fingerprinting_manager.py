import argparse as ap
import datetime as dt
import os
import pickle as pkl
import pprint as pp
import random
import sys
import time
from itertools import batched
from zoneinfo import ZoneInfo

import pandas as pd
from format_data import format_input
from load_data import load_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

print("\n\nStarting hemisphere fingerprinting ...")
print(dt.datetime.now(tz=ZoneInfo("America/New_York")))
print( )

# Set up argument parsers =======

parser = ap.ArgumentParser()

parser.add_argument("--force", "-f", action="store_true",
                    help="Overwrite existing output file, default: False")

parser.add_argument("--join", "-j", default=".",
                    help="Character to join outcomes on, default: '.'")

parser.add_argument("--verbose", "-v", action="store_true",
                    help="Be verbose, default: False")

parser.add_argument("--keep_groups", "-k", nargs="+",
                    metavar=("COL", "VALUE"),
                    help="Values from the 'fold' column to keep. The "
                         "first value is the column ID. The second value is "
                         "the one value to keep. Multiple pairs operate as "
                         "intersection. Must be a list of pairs. "
                         "Must use `--` to end list.")

subparsers = parser.add_subparsers(dest="subcommand")

# Prepare data

prep_parser = subparsers.add_parser('prepare',
                                     help="Given -cifti-convert -to-text "
                                          "outputs and a file with ROI names, "
                                          "create a file for later processing.")

prep_parser.add_argument("--names", "-n",
                         help="File with n names, must match dimensionality "
                              "of input list of pconn files.")

prep_parser.add_argument("output",
                         help="Pickle file to save prepared data to.")

prep_parser.add_argument("text_file", nargs="+",
                         help="An n(x)n correlation matrix from "
                              "-cifti-convert.")

# Training

train_parser = subparsers.add_parser('train',
                                     help="Train and save a model based on "
                                          "training data")

train_parser.add_argument("training_data",
                          help="Pickle with connections (cx_) and identifier "
                               "columns (any other column). All columns in "
                               "the `column` argument must appear in this "
                               "file")

train_parser.add_argument("output",
                          help="Pickle to save the model and columns to.")

train_parser.add_argument("column", nargs="+",
                          help="Column(s) to predict")

train_parser.add_argument("--skip-size-check", action="store_true",
                          help="Do not verify that the input data came from"
                               "one half of a correlation matrix.")

# Testing

test_parser = subparsers.add_parser('test',
                                    help="Given a model and data, predict"
                                         "scores.")

test_parser.add_argument("model",
                         help="Model to use, from this program.")

test_parser.add_argument("test_data",
                          help="Pickle with connections (cx_) and identifier "
                               "columns (any other column). The requested "
                               "columns to predict must appear in this data.")

test_parser.add_argument("output",
                         help="CSV to save the results to, one row per input "
                              "with ground truth, predicted, and all LDs.")

# LOOCV ====

loocv_parser = subparsers.add_parser("loocv",
                                     help="Run a set of LOOCV models and "
                                           "save the results (not models).")

loocv_parser.add_argument("loocv_data",
                          help="Pickle with connections (cx_) and identifier "
                               "columns (any other column). The requested "
                               "columns to predict must appear in this data.")

loocv_parser.add_argument("output")

loocv_parser.add_argument("column", nargs="+",
                          help="Column(s) to predict")

# Null distribution

null_parser = subparsers.add_parser("null",
                                     help="Shuffle the labels and run "
                                          "null repetitions to create an "
                                          "empirical distribution of scalings.")

null_parser.add_argument("null_data",
                          help="Pickle with connections (cx_) and identifier "
                               "columns (any other column). The requested "
                               "columns to predict must appear in this data.")

null_parser.add_argument("output", metavar="CSV",
                         help="CSV file to save the null scalings to.")

null_parser.add_argument("column", nargs="+",
                         help="Column(s) to shuffle and predict.")

null_parser.add_argument("-n", "--repetitions", type=int, default=1000, nargs=1,
                         help="Number of boostraps to run, default=1000. "
                              "Repetition number is saved to the output `n` "
                              "column.")

null_parser.add_argument("-y", "--start", action="store_true",
                         help="Do not prompt the user with the estimated "
                              "execution time, start right away.")

# Parse args =====

args = parser.parse_args()
sc = args.subcommand

if args.verbose:
    pp.pprint(vars(args))

if sc is None:
    parser.print_help()
    sys.exit(1)
else:
    print(f"Subcommand: {sc}")

if args.keep_groups is not None:
    if len(args.keep_groups) % 2 == 0:
        keep_pairs = list(batched(args.keep_groups, 2))
    else:
        print("Must supply at least an even number of arguments to "
              f"'--keep_groups' received: {args.keep_groups}.")
        sys.exit(1)
else:
    keep_pairs = None

if args.verbose:
    print("\nINFO:  Filtering pairs:")
    pp.pprint(keep_pairs)
    print()

# Load data ========

if sc == "train":
    data_to_read = args.training_data
elif sc == "test":
    data_to_read = args.test_data
elif sc == "loocv":
    data_to_read = args.loocv_data
elif sc == "null":
    data_to_read = args.null_data

output_name = args.output

def fit_model(cx, y, test=None):

    # Fit the model
    clf = LinearDiscriminantAnalysis()
    clf.fit(cx, y)

    # Extract results from the model
    ex_var = clf.explained_variance_ratio_

    scalings = pd.DataFrame(clf.scalings_)
    scalings.columns = [f"LD{x}" for x in range(1, scalings.shape[1] + 1)]
    scalings["feature"] = clf.feature_names_in_

    if test is not None:

        y_hat = clf.predict(test)

        # Return pretty scores
        scores = pd.DataFrame(clf.transform(test))
        scores.columns = [f"LD{x}" for x in range(1, scores.shape[1] + 1)]
        scores.reset_index(drop=True, inplace=True)

    else:
        y_hat = scores = None

    return([clf, scalings, ex_var, y_hat, scores])


if not args.force and os.path.exists(output_name):
    print(f"\nERROR: File '{output_name}' already exists! Delete to "
            "continue.\n")
    sys.exit(1)

if sc in ["train", "test", "loocv", "null"]:

    # Load the data as ID and classifier columns
    id, cx = load_data(data_to_read, keep_pairs=keep_pairs)

    # Create merged columns in the case of multiple outcomes
    j = args.join

    # There is no "column" value for the 'test' subcommand: it is included
    # in the serialized object
    if sc in ["train", "loocv", "null"]:

        # Check columns are in data
        pred_cols = args.column
        for col in pred_cols:
            if col not in id.columns:
                print(f"ERROR: Did not find column {col} in the data, exiting")

        # If there is only one column to predict, pass it on
        if len(pred_cols) == 1:
            outcome_name = pred_cols[0]
        else:
            # Otherwise, join the values with an arbitrary column
            print("  -> More than one column to predict")
            outcome_name = j.join(pred_cols)
            outcome_df = id[pred_cols]
            id[outcome_name] = outcome_df.apply(j.join, axis=1)

        print(f"Outcome is: {outcome_name}")

if sc == "train":

    print(f"  Predicting groups from {outcome_name}: {set(id[outcome_name])}")

    # Start up the classifier
    y = id[outcome_name].tolist()

    # Only need the model and explained variance
    clf, _, ex_var, _, _ = fit_model(cx, y)

    print("  Results:")

    ex_var_rnd = [float(round(x, 3)) for x in ex_var]
    print(f"    Explained variance: {ex_var_rnd}")

    acc = clf.score(cx, y)
    print(f"    Accuracy: {acc:.3f}")

    # Save the outcome file name
    result = (outcome_name, clf)

    with open(output_name, "wb") as f:
        pkl.dump(result, f)

    print(f"\nClassifier saved to {output_name}!\n")

elif sc == "test":

    with open(args.model, "rb") as f:
        outcome_name, clf = pkl.load(f)

    outcome_cols = outcome_name.split('.')

    print(f"Predicting column(s) '{outcome_cols}' from the data.")

    # Check that all the columns appear in the data
    if not set(outcome_cols).issubset(id.columns):
        print(f"ERROR: Not all elements of {outcome_cols} appear in the",
              "test data:")
        print(id.columns)
        sys.exit(1)

    # Scores
    y_hat = pd.DataFrame({f"{outcome_name}_predicted": clf.predict(cx)})

    scores = pd.DataFrame(clf.transform(cx))
    scores.columns = [f"LD{x}" for x in range(1, scores.shape[1] + 1)]

    # Reset index so that all three of these run [0, n-1].
    id.reset_index(inplace=True)
    final = pd.concat([id, y_hat, scores], axis=1)

    print("\nResults:\n")
    print(pd.crosstab(final[outcome_name],
                      final[f"{outcome_name}_predicted"]))

    # Calculate accuracy
    n = final.shape[0]
    acc = sum(final[outcome_name] == final[f"{outcome_name}_predicted"]) / n

    print()
    print(f"Accuracy: {acc:.3f}")

    final.to_csv(args.output, index=False)

elif sc == "loocv":

    print(id.shape)
    print(cx.shape)
    n = id.shape[0]

    # Reset the index so it's 0-n instead of maintaing the index before any
    # subsetting with -k
    [x.reset_index(inplace=True) for x in [id, cx]]

    print(f"\nRunning {n} LOOCV repetitions")

    # We are going to append each set of results to a list then concatenate
    # the list of DFs - not optimal
    results = []
    for i in tqdm(range(n)):


        train_id = id.drop(i)
        train_cx = cx.drop(i)

        test_id = id.iloc[[i]]
        test_id.reset_index(drop=True, inplace=True)
        test_cx = cx.iloc[[i]]

        y = train_id[outcome_name].tolist()

        # Start up the classifier
        clf, _, _, y_hat, scores = fit_model(train_cx, y, test=test_cx)
        y_hat_df = pd.DataFrame({f"{outcome_name}_predicted": y_hat})

        # print(scores)

        # One-row pandas DF
        i_final = pd.concat([test_id, y_hat_df, scores], axis=1)
        i_final.reset_index(drop=True, inplace=True)

        # print(i_final)

        results += [i_final]

        # print(i_final)

    result = pd.concat(results)
    result.reset_index(inplace=True, drop=True)

    print("\nResults:\n")

    print(pd.crosstab(result[outcome_name],
                      result[f"{outcome_name}_predicted"]))

    print(result)

    # Calculate accuracy
    acc = sum(result[outcome_name] == result[f"{outcome_name}_predicted"]) / n

    print()
    print(f"Accuracy: {acc:.3f}")

    print(result[outcome_name])
    print(result[f"{outcome_name}_predicted"])

    # Calculate MCC
    mcc = matthews_corrcoef(result[outcome_name],
                            result[f"{outcome_name}_predicted"])

    print(f"MCC:      {mcc:.3f}")

    result.to_csv(args.output, index=False)

elif sc == "null":

    n = args.repetitions[0]

    # Reset the index so it's 0-n instead of maintaing the index before any
    # subsetting with -k
    [x.reset_index(inplace=True, drop=True) for x in [id, cx]]

    results = []

    # Get the y values
    y = id[outcome_name].tolist()
    n_outcomes = len(set(y))
    y_shuffled = random.sample(y, len(y))

    # Time the amount of time it takes to initially fit the model
    start_time = time.perf_counter()
    clf1, scalings1, _, _, _ = fit_model(cx, y_shuffled)
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    total_time = elapsed * n

    # Check accuracy
    acc1 = clf1.score(cx, y)
    print("\nFirst model")
    print(f"Test model accuracy: {acc1:.3f}")
    print(f"Should be close to {1/n_outcomes:.3f}")
    print()

    if total_time < 60:
        print(f"Operation is expected to take {total_time:.0f} seconds")
    elif total_time > 60 and total_time < 3600:
        print(f"Operation is expected to take {total_time/60:.2f} minutes")
    else:
        print(f"Operation is expected to take {total_time/3600:.2f} hours")

    if not args.start:
        user_input = input("Enter 'y' to start: ").lower()
        if user_input != "y":
            sys.exit()

    scalings1["n"] = 0

    print(f"\nRunning {n-1} more null repetitions")
    for i in tqdm(range(1, n)):

        # Keep reshuffling y
        y_shuffled = random.sample(y, len(y))

        # Fit model
        clf, scalings, _, _, _ = fit_model(cx, y_shuffled)
        scalings["n"] = i

        scalings1 = pd.concat([scalings1, scalings], ignore_index=True)
        scalings1["feature"] =scalings1["feature"].astype(object)

    # with open(output_name, "wb") as f:
    #     pkl.dump(scalings1, f)

    scalings1.to_csv(output_name, index=False)

    print(f"Saved {scalings1.shape} output to {output_name}")

elif sc == "prepare":

    files = args.text_file
    formatted_data = format_input(files, "foo", file_with_names=args.names)

    # formatted_data.to_csv(args.output)
    with open(args.output, 'wb') as f:
        pkl.dump(formatted_data, f)

    # For debugging
    formatted_data.to_csv(f"{args.output.replace('pickle', 'csv')}",
                          index=False)