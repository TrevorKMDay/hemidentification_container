import argparse as ap
import pickle as pkl
import os
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import matthews_corrcoef
from load_data import load_data

from tqdm import tqdm

print("Started")

# Set up argument parsers =======

parser = ap.ArgumentParser()

parser.add_argument("--force", "-f", action="store_true",
                    help="Overwrite existing model file.")

parser.add_argument("--join", "-j", default=".",
                    help="Character to join outcomes on, default: .")

parser.add_argument("--keep_groups", "-k", nargs="+",
                    metavar=("COL", "VALUE"),
                    help="Values from the 'fold' column to keep. The "
                         "first value is the column ID. Use `--` to end list.")

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

# Testing

test_parser = subparsers.add_parser('test')

test_parser.add_argument("model",
                         help="Model to use, from this program.")

test_parser.add_argument("test_data")

test_parser.add_argument("output")

test_parser.add_argument("--keep_groups", "-k", nargs="+", default=None,
                          metavar=("COL", "VALUE"),
                          help="Values from the 'fold' column to keep. The "
                                "first value is the column ID.")

# LOOCV ====

loocv_parser = subparsers.add_parser("loocv")

loocv_parser.add_argument("loocv_data")

loocv_parser.add_argument("output")

loocv_parser.add_argument("column", nargs="+",
                          help="Column(s) to predict")


# Parse args =====

args = parser.parse_args()
sc = args.subcommand

if sc is None:
    parser.print_help()
    exit(1)

print(args)
# exit()

if args.keep_groups is not None:
    if len(args.keep_groups) >= 2:
        keep_col = args.keep_groups[0]
        keep_values = args.keep_groups[1:]
    else:
        print("Must supply at least two arguments to '--keep_groups', ",
             f"received: {args.keep_groups}.")
        exit(1)
else:
    keep_col = keep_values = None

# Load data ========

if sc == "train":
    data_to_read = args.training_data
elif sc == "test":
    data_to_read = args.test_data
elif sc == "loocv":
    data_to_read = args.loocv_data

# Create merged columns in the case of multiple outcomes
j = args.join
output_name = args.output

if not args.force and os.path.exists(output_name):
    print(f"\nERROR: File '{output_name}' already exists! Delete to "
            "continue.")
    exit(1)

# Load the data as ID and classifier columns
id, cx = load_data(data_to_read, col=keep_col, col_values=keep_values)

if sc in ["train", "loocv"]:

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

if sc == "train":

    print(f"  Predicting groups from {outcome_name}: {set(id[outcome_name])}")

    # Start up the classifier
    clf = LinearDiscriminantAnalysis()
    y = id[outcome_name].tolist()

    clf.fit(cx, y)
    # print(clf.explained_variance_ratio_)

    print("  Results:")

    ex_var = [float(round(x, 3)) for x in clf.explained_variance_ratio_]
    print(f"    Explained variance: {ex_var}")

    acc = clf.score(cx, y)
    print(f"    Accuracy: {acc:.3f}")

    # Save the outcome file name
    result = (outcome_name, clf)

    with open(output_name, "wb") as f:
        pkl.dump(result, f)

    print(f"\nClassifier saved to {output_name}!")

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
        exit(1)

    y_hat = pd.DataFrame({f"{outcome_name}_predicted": clf.predict(cx)})
    scores = pd.DataFrame(clf.transform(cx))
    scores.columns = [f"LD{x}" for x in range(1, scores.shape[1] + 1)]

    final = pd.concat([id, y_hat, scores], axis=1)

    final.to_csv(args.output, index=False)

elif sc == "loocv":

    n = id.shape[0]

    # Reset the index so it's 0-n instead of maintaing the index before any
    # subsetting with -k
    [x.reset_index(inplace=True) for x in [id, cx]]

    print(f"\nRunning {n} LOOCV repetitions")

    results = []

    for i in tqdm(range(0, n)):

        # print(id)

        train_id = id.drop(i)
        train_cx = cx.drop(i)

        test_id = id.iloc[[i]]
        test_id.reset_index(drop=True, inplace=True)
        test_cx = cx.iloc[[i]]

        # print(test_cx)

        # Start up the classifier
        clf = LinearDiscriminantAnalysis()
        y = train_id[outcome_name].tolist()

        clf.fit(train_cx, y)

        y_hat = pd.DataFrame({f"{outcome_name}_predicted":
                                clf.predict(test_cx)})

        scores = pd.DataFrame(clf.transform(test_cx))
        scores.columns = [f"LD{x}" for x in range(1, scores.shape[1] + 1)]
        scores.reset_index(drop=True, inplace=True)

        i_final = pd.concat([test_id, y_hat, scores], axis=1)
        i_final.reset_index(drop=True, inplace=True)

        results += [i_final]

        # print(i_final)

    result = pd.concat(results)
    result.reset_index(inplace=True, drop=True)

    print("\nResults:\n")

    print(pd.crosstab(result[outcome_name],
                      result[f"{outcome_name}_predicted"]))

    # Calculate accuracy
    acc = sum(result[outcome_name] == result[f"{outcome_name}_predicted"]) / n

    # Calculate MCC
    mcc = matthews_corrcoef(result[outcome_name],
                            result[f"{outcome_name}_predicted"])

    print()
    print(f"Accuracy: {acc:.3f}")
    print(f"MCC:      {mcc:.3f}")

    result.to_csv(args.output)