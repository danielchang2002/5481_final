def main():
  theta = 0.2
  with open("ADH/ADH_MSA.fasta") as f:
    sequences = [line.strip() for line in f if not line.startswith(">")]
  print(sequences)

if __name__ == "__main__":
  main()