# hemidentification_container

This stores the code to train and test connectivity data (anything, really),
using Python `scikit-learn` `LinearDiscriminantAnalysis()`.

The input data must have *n > 0* columns prefixed with `cx_`, these are the
connections. Any column without a `cx_` prefix is assumed to be an identifier
column.

**Important Notes:**

1. The container should be run with the `docker run --rm` flag
    to clean itself up after each run - it does not work persistently like
    Docker expects containers to work.

2. Underscores are used to separate ROI pairs - remove underscores from ROI
    names where possible. The container will check and remove; but if you are
    already tweaking ROI names, then also remove underscores.

## Shared options

- `--force`: Each subcommand has an output file; it is never overwritten unless
    the global force option is passed.

- `--join`: What character to combine the input categories into, default: `.`.
    If there are already periods in the values, change this. For example,
    crossing the `hemi` column with the `diagnosis` column creates a new
    column in the data, `hemi.diagnosis`.

- `--keep_cols`: Sets the name of the column (first argument) and then filters
    the data (train or test) down to the values given in the second+ arguments
    to this flag. Therefore, it requires at least two arguments, e.g.
    `--keep_cols fold 1`.

Note that `keep_cols` only keeps the rows that match all values, e.g.
`-k fold 1 patient hc` keeps all the healthy controls from fold 1. More
advanced subsetting should be added to the data

## Preparing

To prepare a dataset:

    docker run hemidentification prepare [output] pconn.txt [pconn.txt ...]

The container can take Workbench `pconns` that have been converted to text
with `-cifti-convert -to-text`. Reading `pconns` is not included to avoid
having to build `wb_command` into the container.

Each ROI *must* begin with a single-letter code identifying its hemisphere
(i.e., `L` or `R`), but you could use something else or some other division.
The first value is extracted and removed from both ROIs, such that the final
dataset looks something like:

    file    hemi    cx_A_B  cx_A_C  cx_A_D  ...
    f1      L       0.1     -0.1    0.3     ...
    f1      R       0.2     -0.05   0.25    ...
    f2      L       ...     ...     ...     ...
    f2      R       ...     ...     ...     ...
    ...     ...     ...     ...     ...     ...

The `cx_` prefix prefixes *all* connections. Columns without this prefix
are assumed to be identifier columns.

At this step, a pickle is written out only with the file name. If other columns
are needed (e.g., participant group, like control vs. text) should be added
in your own code and then brought forward.

**NB**: You cannot pass a glob to Docker, e.g. `/data/*`, so you have to
accumulate the file names from disk and then pass them with the correct prefix,
e.g.,

    f=$(find data/ -name "*.txt" -exec basename {} \; | sed 's|^|/data/|g')

## Training

To train a model:

    docker run hemidentification train [data] [output] [columns ...]

- `data` is a `pickle` file with the identifier and `cx_` columns described
    above.

- `output` is the name of a `pickle` file that saves the trained model for
    re-use with the `test` subcommand.

- `columns` is a list of one or more columns in `data` to predict.
    Multiple columns are automatically crossed. The selected columns are
    saved with the object so that `test` can automatically read what columns
    to predict from the new data, based on the model.

### Options

- `--skip-size-check`: The code will check that you have the "right" number of
    columns from a lower triangle. If you don't want to check that, use this
    flag. (Not functional yet.)

## Testing

    docker run hemidentification test [model] [data] [output_csv]

- `model` is the `pickle` file created in the previous step.

- `data` is a `pickle` file with the exact same `cx_` columns as in the
    training data. Any other columns are identifier columns.

- `output_csv` is a path to save a file with the identifier columns,
    the predicted class, and the *n - 1* linear discriminant scores.

### LOOCV

The container can also run leave-one-out cross-validation automatically,
saving only the results, not each model. The command is otherwise identical,
but only saves a CSV, no models.

    docker run hemidentification loocv [model] [data] [output_csv]

## Null distribution

One way to identify important scalings (i.e., the relative importance of a
feature/connectionm in predicting the outcome) is to create a null distribution
of scalings by shuffling the outcome labels and running models.

    docker run -it hemidentification null -n N [data] [output csv] [columns]

The default `N` is 1000, which is probably too low for publication, but a
decent starting value for testing.

By default, the code calculates how long the excution will take, and asks you
if you want to start (thus, the `-it` flag is required to interact with the
container in this way).

You can skip this by adding the `-y` flag, however, either `run -t` *or*
`null -y` must be set; it will not save output without one of those two.

    docker run hemidentification null -n N -y [data] [output csv] [columns]

(There is probably a way around this I haven't figured out yet.)

Note that this output is unique in that it saves only the null scalings;
the true scalings are retrievable from the true model created with the
`train` command, and can be accessed through Python or R
`reticulate::py_load_object()`.
