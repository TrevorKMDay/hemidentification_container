# hemidentification_container

This stores the code to train and test connectivity data (anything, really),
using Python `scikit-learn` `LinearDiscriminantAnalysis()`.

The input data must have *n > 0* columns prefixed with `cx_`, these are the
connections. Any column without a `cx_` prefix is assumed to be an identifier
column.

## Shared options

- `--keep_cols` sets the name of the column (first argument) and then filters
    the data (train or test) down to the values given in the second+ arguments
    to this flag. Therefore, it requires at least two arguments, e.g.
    `--keep_cols fold 1`.

## Training

To train a model:

    docker run hemidentification train [data] [output] [columns ...]

- `data` is a `pickle` file with the identifier and `cx_` columns described
    above.

- `output` is the name of a `pickle` file that saves thetrianed model for
    re-use with the `test` subcommand.

- `columns` is a list of one or more columns in `data` to predict.
    Multiple columns are automatically crossed.

### Options

- `--no-skip-size`: The code will check that you have the "right" number of
    columns from a lower triangle. If you don't want to check that, use this
    flag.

- `--join`: What character to combine the input categories into, default: `.`.
    If there are already periods in the values, change this.

## Testing

    docker run hemidentification test [model] [data] [output_csv]

- `model` is the `pickle` file created in the previous step.

- `data` is a `pickle` file with the exact same `cx_` columns as in the
    training data. Any other columns are identifier columns.

- `output_csv` is a path to save a file with the identifier columns,
    the predicted class, and the *n - 1* linear discriminant scores.
