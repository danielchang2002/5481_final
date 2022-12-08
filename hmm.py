import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

THETA = 0.8
DUMMY = False
pseudocount = 0.0000001

class pHMM:
  def __init__(self, sequences):
    # find match states: columns that have less than theta proportion of insertions
    self.match_state = (sequences == "-").mean(axis=0) < THETA

    # iterate through sequences, compute their path through HMM, count emission and transitions

    self.transition_counts = defaultdict(int)
    self.emission_counts = defaultdict(int)
    self.state_counts = defaultdict(int)

    for seq in sequences:
      last_state = "s"
      index = 0
      self.state_counts[last_state] += 1
      for col in range(sequences.shape[1]):
        char = seq[col] != "-"
        match_col = self.match_state[col]
        if char and match_col:
          # match
          index += 1
          new_state = f"m{index}"
          self.transition_counts[(last_state, new_state)] += 1
          last_state = new_state
          self.state_counts[new_state] += 1 + pseudocount
          self.emission_counts[(new_state, seq[col])] += 1
        elif char and not match_col:
          # insertion
          new_state = f"i{index}"
          self.transition_counts[(last_state, new_state)] += 1
          last_state = new_state
          self.state_counts[new_state] += 1 + pseudocount
          self.emission_counts[(new_state, seq[col])] += 1
        elif not char and match_col:
          # deletion
          index += 1
          new_state = f"d{index}"
          self.transition_counts[(last_state, new_state)] += 1
          last_state = new_state
          self.state_counts[new_state] += 1 + pseudocount
      self.transition_counts[(last_state, "e")] += 1
      self.state_counts["e"] += 1

    self.transition_prob = defaultdict(int)
    self.emission_prob = defaultdict(int)

    for t in self.transition_counts:
      self.transition_prob[t] = self.transition_counts[t] / self.state_counts[t[0]]
    
    for e in self.emission_counts:
      self.emission_prob[e] = (self.emission_counts[e] + pseudocount) / self.state_counts[e[0]]


  def score(self, query_seq):
    """
    Returns log prob 
    """
    memo = {}
    best_prev = {}
    num_match_states = self.match_state.sum()
    def score_helper(seq, state):
      # memo case
      if (seq, state) in memo: return memo[(seq, state)]

      score_dict = {}

      # end case
      if state == "e":
        prev_state = f"m{num_match_states}"
        score_dict[prev_state] = score_helper(seq, prev_state) + np.log(self.transition_prob[(prev_state, state)] + pseudocount)
        prev_state = f"i{num_match_states}"
        score_dict[prev_state] = score_helper(seq, prev_state) + np.log(self.transition_prob[(prev_state, state)] + pseudocount)
        prev_state = f"d{num_match_states}"
        score_dict[prev_state] = score_helper(seq, prev_state) + np.log(self.transition_prob[(prev_state, state)] + pseudocount)

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
        e_p = np.log(self.emission_prob[(state, seq[-1])] + pseudocount)

        # start
        if u == 1 and len(seq) == 1:
          prev_state = "s"
          score_dict[prev_state] = e_p + np.log(self.transition_prob[(prev_state, state)])
        
        if len(seq) > 1 and u > 1:
          # match
          prev_state = f"m{u - 1}"
          score_dict[prev_state] = e_p + score_helper(seq[:-1], prev_state) + np.log(self.transition_prob[(prev_state, state)] + pseudocount)

        if len(seq) > 1 and u > 0:
          # insert
          prev_state = f"i{u - 1}"
          score_dict[prev_state] = e_p + score_helper(seq[:-1], prev_state) + np.log(self.transition_prob[(prev_state, state)] + pseudocount)

        if len(seq) > 1:
          # delete
          prev_state = f"d{u - 1}"
          score_dict[prev_state] = e_p + score_helper(seq[:-1], prev_state) + np.log(self.transition_prob[(prev_state, state)] + pseudocount)

      elif state[0] == "i":
        e_p = np.log(self.emission_prob[(state, seq[-1])] + pseudocount)

        # start
        if u == 0 and len(seq) == 1:
          prev_state = "s"
          score_dict[prev_state] = e_p + np.log(self.transition_prob[(prev_state, state)] + pseudocount)

        if len(seq) > 1 and u > 0:
          # match
          prev_state = f"m{u}"
          score_dict[prev_state] = e_p + score_helper(seq[:-1], prev_state) + np.log(self.transition_prob[(prev_state, state)] + pseudocount)

          # insert
        if len(seq) > 1 and u >= 0:
          prev_state = f"i{u}"
          score_dict[prev_state] = e_p + score_helper(seq[:-1], prev_state) + np.log(self.transition_prob[(prev_state, state)] + pseudocount)
      elif state[0] == "d":
        # start
        if u == 1 and len(seq) == 0:
          prev_state = "s"
          score_dict[prev_state] = np.log(self.transition_prob[(prev_state, state)] + pseudocount)

        if seq != "" and u > 1:
          # match
          prev_state = f"m{u - 1}"
          score_dict[prev_state] = score_helper(seq, prev_state) + np.log(self.transition_prob[(prev_state, state)] + pseudocount)

        if u > 1:
          # delete
          prev_state = f"d{u - 1}"
          score_dict[prev_state] = score_helper(seq, prev_state) + np.log(self.transition_prob[(prev_state, state)] + pseudocount)
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

    return s

def main():
  with open("ADH/train.fasta") as f:
    adh_train_sequences = np.array([list(line.strip()) for line in f if not line.startswith(">")])
  with open("ACDH/train.fasta") as f:
    acdh_train_sequences = np.array([list(line.strip()) for line in f if not line.startswith(">")])
  with open("ADH/test.fasta") as f:
    adh_test_sequences = [line.strip() for line in f if not line.startswith(">")]
  with open("ACDH/test.fasta") as f:
    acdh_test_sequences = [line.strip() for line in f if not line.startswith(">")]

  print(np.mean([len(seq) for seq in adh_train_sequences]))
  print(np.mean([len(seq) for seq in acdh_train_sequences]))
  print(np.mean([len(seq) for seq in adh_test_sequences]))
  print(np.mean([len(seq) for seq in acdh_test_sequences]))

  adh_pHMM = pHMM(adh_train_sequences)
  acdh_pHMM = pHMM(acdh_train_sequences)

  adh_pHMM_adh = []
  acdh_pHMM_adh = []
  adh_pHMM_acdh = []
  acdh_pHMM_acdh = []

  for adh_test in adh_test_sequences:
    n = len(adh_test)
    adh_pHMM_adh.append(adh_pHMM.score(adh_test))
    acdh_pHMM_adh.append(acdh_pHMM.score(adh_test))

  for acdh_test in acdh_test_sequences:
    n = len(acdh_test)
    adh_pHMM_acdh.append(adh_pHMM.score(acdh_test))
    acdh_pHMM_acdh.append(acdh_pHMM.score(acdh_test))

  plt.hist(adh_pHMM_adh, label="alcohol dehydrogenase")
  plt.hist(adh_pHMM_acdh, label="acetaldehyde dehydrogenase")
  plt.xlabel("Log Likelihood of alcohol dehydrogenase Profile HMM on holdout genes")
  plt.ylabel("Count")
  plt.legend()
  plt.show()

  plt.hist(acdh_pHMM_adh, label="alcohol dehydrogenase")
  plt.hist(acdh_pHMM_acdh, label="acetaldehyde dehydrogenase")
  plt.xlabel("Log Likelihood of acetaldehyde dehydrogenase Profile HMM on holdout genes")
  plt.ylabel("Count")
  plt.legend()
  plt.show()

if __name__ == "__main__":
  main()