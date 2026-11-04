import os
import datetime as dt

def is_output_newer(in_file, out_file):
    """
    Return True if the second file is newer, and print a message for the user.
    """

    in_ts = os.path.getmtime(in_file)
    out_ts = os.path.getmtime(out_file)

    f1 = in_file if in_ts > out_ts else out_file
    f2 = out_file if in_ts > out_ts else in_file

    print(f"INFO: '{f1}' is newer "
          f"({dt.datetime.fromtimestamp(max(in_ts, out_ts))}) than '{f2}' "
          f"({dt.datetime.fromtimestamp(min(in_ts, out_ts))}).")

    # Return whether the output file is newer
    return(out_ts > in_ts)