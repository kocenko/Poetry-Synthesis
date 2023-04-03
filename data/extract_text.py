#-----------------------------------------------------------------
# Module used only to extract data of interest from the csv file.
#-----------------------------------------------------------------

import csv
import os


def extract(in_file_name: str, out_file_name: str, overwrite: bool = False):
    '''Reads from csv file and writes the content of one row to the text file
    
    Args:
        in_file_name (str): path of the file to read from
        out_file_name (str): path of the file to write to
    '''

    dir_path = os.path.dirname(__file__)
    in_file_name = os.path.join(dir_path, in_file_name)
    out_file_name = os.path.join(dir_path, out_file_name)

    assert os.path.isfile(in_file_name), "Could not find an input file. Make sure it is in the same folder as the script"
    assert (os.path.isfile(out_file_name) and overwrite) or not os.path.isfile(out_file_name), ("Output file already exists. " 
                                                                                                "Change function parameter to overwrite")

    with open(in_file_name, mode='r', encoding="utf8") as input_file:
        reader = csv.reader(input_file, delimiter=',')
        next(reader, None)  # To skip header 
        with open(out_file_name, 'w', encoding="utf8") as output_file:
            for row in reader:
                output_file.write(row[1] + '\n')


if __name__ == "__main__":

    input_file_name = "preprocessed_data.csv"
    output_file_name = "poe_data.txt"

    extract(input_file_name, output_file_name, True)