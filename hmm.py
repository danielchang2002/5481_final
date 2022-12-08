import numpy as np
from collections import defaultdict
import pdb

THETA = 0.35
TEST = True
pseudocount = 0.01

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

  num_match_states = match_state.sum()

  test_seq = "ACAFDEAF"

  # return log prob
  def score(query_seq):
    memo = {}
    best_prev = {}
    def score_helper(seq, state):
      # memo case
      if (seq, state) in memo: return memo[(seq, state)]

      score_dict = {}

      # end case
      if state == "e":
        prev_state = f"m{num_match_states}"
        score_dict[prev_state] = score_helper(seq, prev_state) + np.log(transition_prob[(prev_state, state)] + pseudocount)
        prev_state = f"i{num_match_states}"
        score_dict[prev_state] = score_helper(seq, prev_state) + np.log(transition_prob[(prev_state, state)] + pseudocount)
        prev_state = f"d{num_match_states}"
        score_dict[prev_state] = score_helper(seq, prev_state) + np.log(transition_prob[(prev_state, state)] + pseudocount)

        best_score = -np.inf
        best_prev_state = None
        for prev_state in score_dict:
          if score_dict[prev_state] > best_score:
            best_score = score_dict[prev_state]
            best_prev_state = prev_state
        
        memo[(seq, state)] = best_score
        best_prev[(seq, state)] = best_prev_state

        return best_score

      u = int(state[1])

      if state[0] == "m":
        e_p = np.log(emission_prob[(state, seq[-1])] + pseudocount)

        # start
        if u == 1 and len(seq) == 1:
          prev_state = "s"
          score_dict[prev_state] = e_p + np.log(transition_prob[(prev_state, state)])
        
        if len(seq) > 1 and u > 1:
          # match
          prev_state = f"m{u - 1}"
          score_dict[prev_state] = e_p + score_helper(seq[:-1], prev_state) + np.log(transition_prob[(prev_state, state)] + pseudocount)

        if len(seq) > 1 and u > 0:
          # insert
          prev_state = f"i{u - 1}"
          score_dict[prev_state] = e_p + score_helper(seq[:-1], prev_state) + np.log(transition_prob[(prev_state, state)] + pseudocount)

        if len(seq) > 1:
          # delete
          prev_state = f"d{u - 1}"
          score_dict[prev_state] = e_p + score_helper(seq[:-1], prev_state) + np.log(transition_prob[(prev_state, state)] + pseudocount)

      elif state[0] == "i":
        e_p = np.log(emission_prob[(state, seq[-1])] + pseudocount)

        # start
        if u == 0 and len(seq) == 1:
          prev_state = "s"
          score_dict[prev_state] = e_p + np.log(transition_prob[(prev_state, state)] + pseudocount)

        if len(seq) > 1 and u > 0:
          # match
          prev_state = f"m{u}"
          score_dict[prev_state] = e_p + score_helper(seq[:-1], prev_state) + np.log(transition_prob[(prev_state, state)] + pseudocount)

          # insert
        if len(seq) > 1 and u >= 0:
          prev_state = f"i{u}"
          score_dict[prev_state] = e_p + score_helper(seq[:-1], prev_state) + np.log(transition_prob[(prev_state, state)] + pseudocount)
      elif state[0] == "d":
        # start
        if u == 1 and len(seq) == 0:
          prev_state = "s"
          score_dict[prev_state] = np.log(transition_prob[(prev_state, state)] + pseudocount)

        if seq != "" and u > 1:
          # match
          prev_state = f"m{u - 1}"
          score_dict[prev_state] = score_helper(seq, prev_state) + np.log(transition_prob[(prev_state, state)] + pseudocount)

        if u > 1:
          # delete
          prev_state = f"d{u - 1}"
          score_dict[prev_state] = score_helper(seq, prev_state) + np.log(transition_prob[(prev_state, state)] + pseudocount)
      else:
        print("bruh error")
        return

      best_score = -np.inf
      best_prev_state = None
      for prev_state in score_dict:
        if score_dict[prev_state] > best_score:
          best_score = score_dict[prev_state]
          best_prev_state = prev_state
      
      memo[(seq, state)] = best_score
      best_prev[(seq, state)] = best_prev_state

      return best_score

    s = score_helper(query_seq, "e")
    return s, best_prev

  s, best_prev = score(test_seq)
  print(s)

  curr = "e"
  seq = test_seq

  while True:
    print(seq, curr)
    if (seq, curr) not in best_prev: break
    curr = best_prev[(seq, curr)]
    if curr[0] == "m" or curr[0] == "i":
      seq = seq[:-1]


if __name__ == "__main__":
  main()