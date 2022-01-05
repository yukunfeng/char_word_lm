import argparse


def main(args):
  with open(args.input_path, 'r') as fh:
    length2sents = {"1_15":[], "15_30":[], "30_":[]}
    longest = 0
    for line in fh:
      line = line.strip()
      if line == "":
        continue
      tokens = line.split()
      longest = max(len(tokens), longest)
      if len(tokens) <= 15:
        length2sents["1_15"].append(line)
      elif len(tokens) <= 20:
        length2sents["15_30"].append(line)
      else:
        length2sents["30_"].append(line)
    print(f"longest: {longest}")
    print(f"1_15: {len(length2sents['1_15'])}")
    print(f"15_30: {len(length2sents['15_30'])}")
    print(f"30_: {len(length2sents['30_'])}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('--input_path', required=True, type=str)
  args = parser.parse_args()
  main(args)
