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
from is_output_newer import is_output_newer
from fit_model import fit_model

from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

# This is updated externally by `build.sh`
VERSION="0.5.0"
BUILD_DATE="Wed Nov  4 10:00:34 EST 2026"

class SmartFormatter(ap.HelpFormatter):
    # Source - https://stackoverflow.com/a/22157136
    # Posted by Anthon, modified by community.
    # Retrieved 2026-10-21, License - CC BY-SA 3.0
    def _split_lines(self, text, width):
        """Allow forced line breaks in argparse help text."""
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return ap.HelpFormatter._split_lines(self, text, width)

# Set up argument parsers =======

parser = ap.ArgumentParser(formatter_class=SmartFormatter)

parser.add_argument("--file_conflict", "-f",
                    choices=["strict", "compare", "force"],
                    default="strict",
                    help="R|strict:  Do not replace an existing file\n"
                         "compare: Rerun if input data is newer\n"
                         "force:   Always recreate output.")

parser.add_argument("--join", "-j", default=".",
                    help="Character to join outcomes on, default: '.'")

parser.add_argument("--verbose", "-v", action="store_true",
                    help="Be verbose, default: False")

parser.add_argument("--keep_groups", "-k", nargs="+",
                    metavar=("COL", "VALUE"),
                    help="R|Values from the 'fold' column to keep.\n"
                         "The first value is the column ID.\n"
                         "The second value is the one value to keep.\n"
                         "Check inequality by prefixing value with '!'.\n"
                         "Multiple pairs operate as intersection.\n"
                         "Must be a list of pairs.\n"
                         "Must use `--` to end list.")

subparsers = parser.add_subparsers(dest="subcommand")

# Prepare data

prep_parser = subparsers.add_parser('version',
                                     help="Report version and exit.")


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

null_parser.add_argument("output", metavar="output",
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

# Give some cleaner names
args = parser.parse_args()
sc = args.subcommand
verbose = args.verbose

if verbose:
    print("Verbose output:")
    pp.pprint(vars(args))
    print()

if sc == "version":
    # If only 'version' print info and quit with zero exit status
    print(f"Version:    {VERSION}")
    print(f"Build date: {BUILD_DATE}")
    print("GitHub:     "
          "https://github.com/TrevorKMDay/hemidentification_container")
    sys.exit(0)
elif sc is None:
    parser.print_help()
    sys.exit(1)

print("Starting hemisphere fingerprinting ...")
print(f"Version: {VERSION}")
print(f"Start: {dt.datetime.now(tz=ZoneInfo('America/New_York'))}")
print()
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

clobber = args.file_conflict

# Load data ========

if sc == "train":
    data_to_read = args.training_data
elif sc == "test":
    data_to_read = args.test_data
elif sc == "loocv":
    data_to_read = args.loocv_data
elif sc == "null":
    data_to_read = args.null_data

# Give a cleaner name
output_name = args.output

# This is the same for train/test/loocv/null
if data_to_read is not None:
    if not "pickle" in data_to_read:
        print(f"ERROR: Unknown format: {data_to_read}, should be '.pickle'.")

if sc in ["test", "loocv", "null"]:
    if ".csv" not in output_name:
        print(f"ERROR: Unexpected format: {output_name} should be '.csv' " +
              "for null.")
        sys.exit(1)

# Check for zipped csvs
output_name_zip = output_name.replace(".csv", ".zip")

if os.path.exists(output_name_zip):

    print(f"INFO: Found zipped file with same name: {output_name_zip}")

    # Compare against the ZIP file, unless the CSV exists, and is newer
    # If the CSV doesn't exist, it's been zipped to save space, and check
    # against the ZIP file modification times.
    output_comparison_file = output_name_zip
    if os.path.exists(output_name):
        # If the ZIP is older than the CSV
        if not is_output_newer(output_name, output_name_zip):
            output_comparison_file = output_name
            print(f"INFO: Zipped file {output_name_zip} is out of date!")
        else:
            print(f"INFO: Zipped file {output_name_zip} is up-to-date, "
                  "checking it against input timestamps")
        print()


# Compare the input/output timestamps
if os.path.exists(output_comparison_file):

    # If strict, and the output exists, don't clobber it
    if clobber == "strict":

        print(f"\nERROR: File '{output_comparison_file}' already exists! "
              "Delete or use "
              "'--file_conflict compare' or '--file_conflict force' to "
              "continue.\n")
        sys.exit(1)

    # Check timestamps if 'compare' is set, and exit if the output is newer
    # than the input
    elif clobber == "compare":

        out_nt_in = is_output_newer(data_to_read, output_comparison_file)

        # Only the test module has a second input, so if we're not testing,
        # pass True to the all() check below so it passes unless the model
        # is newer than the output.
        model_nt_in = is_output_newer(args.model, output_comparison_file) \
            if sc == "test" else True

        # Check that all conditions are met, and exit with a good exit status
        # if the output is newer than the inputs.
        if all([out_nt_in, model_nt_in]):
            print(f"INFO: The output file '{output_comparison_file}' is newer "
                  "than the input(s), not doing anything.\n")
            print("Exiting!\n")
            sys.exit(0)
        else:
            print("INFO: Input(s) are newer than output, starting ...\n")

    # If forcing, don't check anything and keep going
    elif clobber == "force":

        print(f"WARNING: Overwriting output file {output_comparison_file} "
              "without checking.\n")

sys.exit()

# sys.exit()

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
    print(clf.explained_variance_ratio_)

    # Round this for echoing
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

    # print(f"Outcome name: {outcome_name}")
    # print(f"Classifier: {clf}")

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
    print("Scores:")
    print(clf.transform(cx))

    scores.columns = [f"LD{x}" for x in range(1, scores.shape[1] + 1)]

    # print(scores)

    # Reset index so that all three of these run [0, n-1].
    id = id.reset_index(drop=True)
    final = pd.concat([id, y_hat, scores], axis=1)

    # Need to create joined colum
    if len(outcome_cols) > 1:
        f_outcome_cols = final[outcome_cols]
        final[outcome_name] = f_outcome_cols.apply(j.join, axis=1)

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

    # print(id.shape)
    # print(cx.shape)
    n = id.shape[0]

    # Reset the index so it's 0-n instead of maintaing the index before any
    # subsetting with -k
    # [x.reset_index(inplace=True) for x in [id, cx]]
    id = id.reset_index(drop=True)
    cx = cx.reset_index(drop=True)

    print(f"\nRunning {n} LOOCV repetitions")

    # We are going to append each set of results to a list then concatenate
    # the list of DFs - not optimal
    results = []
    for i in tqdm(range(n)):


        train_id = id.drop(i)
        train_cx = cx.drop(i)

        test_id = id.iloc[[i]]
        test_id = test_id.reset_index(drop=True)
        test_cx = cx.iloc[[i]]

        y = train_id[outcome_name].tolist()

        # Start up the classifier
        clf, _, _, y_hat, scores = fit_model(train_cx, y, test=test_cx)
        y_hat_df = pd.DataFrame({f"{outcome_name}_predicted": y_hat})

        # print(scores)

        # One-row pandas DF
        i_final = pd.concat([test_id, y_hat_df, scores], axis=1)
        i_final = i_final.reset_index(drop=True)

        # print(i_final)

        results += [i_final]

        # print(i_final)

    result = pd.concat(results)
    result = result.reset_index(drop=True)

    print("\nResults:\n")

    print(pd.crosstab(result[outcome_name],
                      result[f"{outcome_name}_predicted"]))

    # print(result)

    # Calculate accuracy
    acc = sum(result[outcome_name] == result[f"{outcome_name}_predicted"]) / n

    print()
    print(f"Accuracy: {acc:.3f}")

    # print(result[outcome_name])
    # print(result[f"{outcome_name}_predicted"])

    # Calculate MCC
    mcc = matthews_corrcoef(result[outcome_name],
                            result[f"{outcome_name}_predicted"])

    print(f"MCC:      {mcc:.3f}")

    result.to_csv(args.output, index=False)

elif sc == "null":

    n = args.repetitions[0]

    # Reset the index so it's 0-n instead of maintaing the index before any
    # subsetting with -k
    # [x.reset_index(inplace=True, drop=True) for x in [id, cx]]
    id = id.reset_index(drop=True)
    cx = cx.reset_index(drop=True)

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