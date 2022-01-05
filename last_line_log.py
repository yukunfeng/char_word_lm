import sys
import ast
import re

if __name__ == "__main__":
    fh_list = []
    if len(sys.argv) >= 2:
        extra_params = sys.argv[1:]
        for file_path in extra_params:
            fh = open(file_path, 'r')
            fh_list.append(fh)
    else:
        fh = sys.stdin
        fh_list.append(fh)

    for fh in fh_list:
        for line in fh:
            line = line.strip()
            if line == "":
                continue
            x = ast.literal_eval(line)
            x = [f"{val:.0f}" for val in x]
            print("\t".join(x))
    for fh in fh_list:
        fh.close()
