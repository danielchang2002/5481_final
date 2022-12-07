import numpy as np
from collections import defaultdict

THETA = 0.35
TEST = True

def main():
  if TEST:
    sequences = np.array([
      ["A", "C", "D", "E", "F", "A", "C", "A", "D", "F",],
      ["A", "F", "D", "A", "-", "-", "-", "C", "C", "F",],
      ["A", "-", "-", "E", "F", "D", "-", "F", "D", "C",],
      ["A", "C", "A", "E", "F", "-", "-", "A", "-", "C",],
      ["A", "D", "D", "E", "F", "A", "A", "A", "D", "F",],
    ])
  else:
    with open("ADH/ADH_MSA.fasta") as f:
      sequences = np.array([list(line.strip()) for line in f if not line.startswith(">")])

  # find match states: columns that have less than theta proportion of insertions
  match_state = (sequences == "-").mean(axis=0) < THETA

  # iterate through sequences, compute their path through HMM, count emission and transitions

  transition_counts = defaultdict(int)
  emission_counts = defaultdict(int)
  state_counts = defaultdict(int)

  for seq in sequences:
    last_state = "s"
    index = 0
    state_counts[last_state] += 1
    for col in range(sequences.shape[1]):
      char = seq[col] != "-"
      match_col = match_state[col]
      if char and match_col:
        # match
        index += 1
        new_state = f"m{index}"
        transition_counts[(last_state, new_state)] += 1
        last_state = new_state
        state_counts[new_state] += 1
        emission_counts[(new_state, seq[col])] += 1
      elif char and not match_col:
        # insertion
        new_state = f"i{index}"
        transition_counts[(last_state, new_state)] += 1
        last_state = new_state
        state_counts[new_state] += 1
        emission_counts[(new_state, seq[col])] += 1
      elif not char and match_col:
        # deletion
        index += 1
        new_state = f"d{index}"
        transition_counts[(last_state, new_state)] += 1
        last_state = new_state
        state_counts[new_state] += 1
    transition_counts[(last_state, "e")] += 1
    state_counts["e"] += 1

  transition_prob = defaultdict(int)
  emission_prob = defaultdict(int)
  for t in transition_counts:
    transition_prob[t] = transition_counts[t] / state_counts[t[0]]
  
  for e in emission_counts:
    emission_prob[e] = emission_counts[e] / state_counts[e[0]]

  print(transition_prob)
  print(emission_prob)

  num_match_states = match_state.sum()

  # construct transition probability matrix
  # n = 3 + 3 * num_match_states
  # transition_prob = np.zeros((n, n))

  # def get_idx(state):
  #   if state == "s": return 0
  #   if state == "e": return n - 1
  #   state_type, num = state[0], int(state[1])
  #   if state_type == "i":
  #     return 1 + num * 3
  #   if state_type == "m":
  #     return num * 3 - 1
  #   if state_type == "d":
  #     return num * 3

  # for transition in transition_counts:
  #   before, after = transition
  #   before_idx, after_idx = get_idx(before), get_idx(after)

  #   # TODO add pseudocount
  #   prob = transition_counts[transition] / sequences.shape[0]
  #   transition_prob[before_idx][after_idx] = prob

  # print(transition_prob)


if __name__ == "__main__":
  main()