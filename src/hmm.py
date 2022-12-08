import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

THETA = 0.35
pseudocount = 2
DUMMY_TEST = False
np.random.seed(42)
TRAIN_TEST_SPLIT = 0.8

class pHMM:
  def __init__(self, sequences):
    # find match states: columns that have less than theta proportion of insertions
    self.match_state = (sequences == "-").mean(axis=0) < THETA
    self.num_match_states = self.match_state.sum()
    self.unique_chars = np.unique([char for char in sequences.flatten() if char != "-"])

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
          self.state_counts[new_state] += 1
          self.emission_counts[(new_state, seq[col])] += 1
        elif char and not match_col:
          # insertion
          new_state = f"i{index}"
          self.transition_counts[(last_state, new_state)] += 1
          last_state = new_state
          self.state_counts[new_state] += 1
          self.emission_counts[(new_state, seq[col])] += 1
        elif not char and match_col:
          # deletion
          index += 1
          new_state = f"d{index}"
          self.transition_counts[(last_state, new_state)] += 1
          last_state = new_state
          self.state_counts[new_state] += 1
      self.transition_counts[(last_state, "e")] += 1
      self.state_counts["e"] += 1

    states = ["s", "i0"]

    for i in range(1, self.num_match_states + 1):
      states.append(f"m{i}")
      states.append(f"d{i}")
      states.append(f"i{i}")

    states.append("e")

    # inflate state counts
    for s in states:
      self.state_counts[s] += pseudocount

    # inflate emission counts
    for s in states:
      if s[0] != "m" and s[0] != "i":
        continue
      for char in self.unique_chars:
        self.emission_counts[(s, char)] += 1

    self.states = states

    # inflate transition counts

    # inflate s transition
    self.transition_counts[("s", "i0")] += 1
    self.transition_counts[("s", "m1")] += 1
    self.transition_counts[("s", "d1")] += 1

    # inflate i transitions
    for i in range(self.num_match_states):
      self.transition_counts[(f"i{i}", f"i{i}")] += 1
      self.transition_counts[(f"i{i}", f"m{i + 1}")] += 1
      self.transition_counts[(f"i{i}", f"d{i + 1}")] += 1

    # inflate m and d transitions
    for i in range(1, self.num_match_states + 1):
      self.transition_counts[(f"m{i}", f"i{i}")] += 1
      self.transition_counts[(f"d{i}", f"i{i}")] += 1
      if i != self.num_match_states:
        self.transition_counts[(f"m{i}", f"m{i + 1}")] += 1
        self.transition_counts[(f"m{i}", f"d{i + 1}")] += 1
        self.transition_counts[(f"d{i}", f"m{i + 1}")] += 1
        self.transition_counts[(f"d{i}", f"d{i + 1}")] += 1

    # inflate e transition
    self.transition_counts[("m8", "e")] += 1
    self.transition_counts[("d8", "e")] += 1
    self.transition_counts[("i8", "e")] += 1

    self.transition_prob = defaultdict(int)
    self.emission_prob = defaultdict(int)

    for t in self.transition_counts:
      self.transition_prob[t] = np.log(self.transition_counts[t] / self.state_counts[t[0]])
    
    for e in self.emission_counts:
      self.emission_prob[e] = np.log(self.emission_counts[e] / self.state_counts[e[0]])

  def score(self, query_seq):
    """
    Returns log prob of optimal path of query_seq through the profile HMM.
    Uses the viterbi algorithm
    """
    # ------------------------------ testing ----------------------------
    if DUMMY_TEST:
      # Use pevzner textbook example
      goal = self.transition_prob[("s", "m1")] + \
        self.emission_prob[("m1", "A")] + \
        self.transition_prob[("m1", "m2")] + \
        self.emission_prob[("m2", "C")] + \
        self.transition_prob[("m2", "i2")] + \
        self.emission_prob[("i2", "A")] + \
        self.transition_prob[("i2", "i2")] + \
        self.emission_prob[("i2", "F")] + \
        self.transition_prob[("i2", "m3")] + \
        self.emission_prob[("m3", "D")] + \
        self.transition_prob[("m3", "m4")] + \
        self.emission_prob[("m4", "E")] + \
        self.transition_prob[("m4", "d5")] + \
        self.transition_prob[("d5", "m6")] + \
        self.emission_prob[("m6", "A")] + \
        self.transition_prob[("m6", "d7")] + \
        self.transition_prob[("d7", "m8")] + \
        self.emission_prob[("m8", "F")] + \
        self.transition_prob[("m8", "e")]
      print(goal)
    # ------------------------------ testing ----------------------------

    dp = {}
    # backpointer = {}

    def score_helper(seq, state):
      # memoization
      if (seq, state) in dp: return dp[(seq, state)]

      # error cases
      if (state == "s" and seq != "") or \
        (((state[0] == "m") or (state[0] == "i")) and seq == "") or \
        ((state[0] == "i") and int(state[1:]) < 0) or \
        ((state[0] == "m" or state[0] == "d") and int(state[1:]) < 1):
        dp[(seq, state)] = -np.inf
        return -np.inf
      
      # base case
      if state == "s":
        assert(seq == "")
        dp[(seq, state)] = 0
        return 0

      # compute emission probability
      e_p = self.emission_prob[(state, seq[-1])] if state[0] == "m" or state[0] == "i" else 0

      u = int(state[1:])

      if state[0] == "m":
        scores = e_p + np.array([
          score_helper(seq[:-1], f"m{u - 1}") + self.transition_prob[(f"m{u - 1}", state)],
          score_helper(seq[:-1], f"i{u - 1}") + self.transition_prob[(f"i{u - 1}", state)],
          score_helper(seq[:-1], f"d{u - 1}") + self.transition_prob[(f"d{u - 1}", state)],
          (score_helper(seq[:-1], "s") + self.transition_prob[("s", state)]) if u == 1 else -np.inf
        ])
        dp[(seq, state)] = np.max(scores)

        prev_states = [f"m{u - 1}", f"i{u - 1}", f"d{u - 1}", "s"]
        # backpointer[(seq, state)] = (seq[:-1], prev_states[np.argmax(scores)])

        return np.max(scores)
      elif state[0] == "i":
        scores = e_p + np.array([
          score_helper(seq[:-1], f"m{u}") + self.transition_prob[(f"m{u}", state)],
          score_helper(seq[:-1], f"i{u}") + self.transition_prob[(f"i{u}", state)],
          (score_helper(seq[:-1], "s") + self.transition_prob[("s", state)]) if u == 0 else -np.inf
        ])
        dp[(seq, state)] = np.max(scores)

        prev_states = [f"m{u}", f"i{u}", "s"]
        # backpointer[(seq, state)] = (seq[:-1], prev_states[np.argmax(scores)])

        return np.max(scores)

      elif state[0] == "d":
        scores = np.array([
          score_helper(seq, f"m{u - 1}") + self.transition_prob[(f"m{u - 1}", state)],
          score_helper(seq, f"d{u - 1}") + self.transition_prob[(f"d{u - 1}", state)],
          (score_helper(seq, "s") + self.transition_prob[("s", state)]) if u == 1 else -np.inf
        ])
        dp[(seq, state)] = np.max(scores)

        prev_states = [f"m{u - 1}", f"d{u - 1}", "s"]
        # backpointer[(seq, state)] = (seq, prev_states[np.argmax(scores)])

        return np.max(scores)
      else:
        return "bruh error"

    scores = np.array([
      score_helper(query_seq, f"m{self.num_match_states}") + self.transition_prob[(f"m{self.num_match_states}", "e")],
      score_helper(query_seq, f"i{self.num_match_states}") + self.transition_prob[(f"i{self.num_match_states}", "e")],
      score_helper(query_seq, f"d{self.num_match_states}") + self.transition_prob[(f"d{self.num_match_states}", "e")]
    ])

    prev_states = [f"m{self.num_match_states}", f"d{self.num_match_states}", f"d{self.num_match_states}"]
    # backpointer[(query_seq, "e")] = (query_seq, prev_states[np.argmax(scores)])
    return np.max(scores)

def main():
  # ------------------------------ testing ----------------------------
  if DUMMY_TEST:
    # Use pevzner textbook example
    sequences = np.array([
        ["A", "C", "D", "E", "F", "A", "C", "A", "D", "F",],
        ["A", "F", "D", "A", "-", "-", "-", "C", "C", "F",],
        ["A", "-", "-", "E", "F", "D", "-", "F", "D", "C",],
        ["A", "C", "A", "E", "F", "-", "-", "A", "-", "C",],
        ["A", "D", "D", "E", "F", "A", "A", "A", "D", "F",],
    ])

    p = pHMM(sequences)

    # Use pevzner textbook example
    test = "ACAFDEAF"

    score, backpointer = p.score(test)
    print(score)

    # compute optimal path using backpointers
    curr = "e"
    query_seq = test
    while True:
      print(curr, "->", backpointer[(query_seq, curr)][1])
      query_seq, curr = backpointer[(query_seq, curr)]
      if (query_seq, curr) not in backpointer: break
    return
  # ------------------------------ testing ----------------------------
  
  # load dataset up
  with open("../data/ADH/ADH_MSA.fasta") as f:
    adh = np.array([list(line.strip()) for line in f if not line.startswith(">")])
  with open("../data/ACDH/ACDH_MSA.fasta") as f:
    acdh = np.array([list(line.strip()) for line in f if not line.startswith(">")])

  # train test split
  adh_train_index = np.zeros((adh.shape[0])) == 1
  adh_train_index[np.random.choice(adh.shape[0], replace=False, size=int(adh.shape[0] * TRAIN_TEST_SPLIT))] = True

  acdh_train_index = np.zeros((acdh.shape[0])) == 1
  acdh_train_index[np.random.choice(acdh.shape[0], replace=False, size=int(acdh.shape[0] * TRAIN_TEST_SPLIT))] = True

  adh_train = adh[adh_train_index]
  acdh_train = acdh[acdh_train_index]

  adh_test = ["".join([char for char in a if char != "-"]) for a in adh[~adh_train_index]]
  acdh_test = ["".join([char for char in a if char != "-"]) for a in acdh[~acdh_train_index]]

  # train models
  print("training adh pHMM")
  adh_pHMM = pHMM(adh_train)
  print("training acdh pHMM")
  acdh_pHMM = pHMM(acdh_train)

  # compute scores
  adh_pHMM_adh = []
  adh_pHMM_acdh = []
  acdh_pHMM_adh = []
  acdh_pHMM_acdh = []

  print("Computing scores on adh test set")
  for adh_test_seq in tqdm(adh_test):
    adh_pHMM_adh.append(adh_pHMM.score(adh_test_seq))
    acdh_pHMM_adh.append(acdh_pHMM.score(adh_test_seq))

  print("Computing scores on acdh test set")
  for acdh_test_seq in tqdm(acdh_test):
    adh_pHMM_acdh.append(adh_pHMM.score(acdh_test_seq))
    acdh_pHMM_acdh.append(acdh_pHMM.score(acdh_test_seq))

  with open("../output/holdout_scores/adh_pHMM_adh", "w") as f:
    for item in adh_pHMM_adh:
      f.write(f"{item}\n")
      
  with open("../output/holdout_scores/acdh_pHMM_adh", "w") as f:
    for item in acdh_pHMM_adh:
      f.write(f"{item}\n")

  with open("../output/holdout_scores/adh_pHMM_acdh", "w") as f:
    for item in adh_pHMM_acdh:
      f.write(f"{item}\n")

  with open("../output/holdout_scores/acdh_pHMM_acdh", "w") as f:
    for item in acdh_pHMM_acdh:
      f.write(f"{item}\n")

  


if __name__ == "__main__":
  main()