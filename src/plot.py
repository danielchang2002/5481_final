import matplotlib.pyplot as plt

def main():
  # load scores
  adh_pHMM_adh = []
  adh_pHMM_acdh = []
  acdh_pHMM_adh = []
  acdh_pHMM_acdh = []

  with open("../output/holdout_scores/adh_pHMM_adh") as f:
    for item in f:
      adh_pHMM_adh.append(float(item))
      
  with open("../output/holdout_scores/acdh_pHMM_adh") as f:
    for item in f:
      acdh_pHMM_adh.append(float(item))

  with open("../output/holdout_scores/adh_pHMM_acdh") as f:
    for item in f:
      adh_pHMM_acdh.append(float(item))

  with open("../output/holdout_scores/acdh_pHMM_acdh") as f:
    for item in f:
      acdh_pHMM_acdh.append(float(item))


  plt.hist(adh_pHMM_adh, label="alcohol dehydrogenase", bins=20)
  plt.hist(adh_pHMM_acdh, label="acetaldehyde dehydrogenase", bins=20)
  plt.xlabel("Log Likelihood of alcohol dehydrogenase Profile HMM on holdout genes", fontsize=20)
  plt.ylabel("Count", fontsize=20)
  plt.legend(prop={'size': 24})
  plt.savefig("../output/figures/ADH_pHMM.png")
  plt.show()

  plt.hist(acdh_pHMM_adh, label="alcohol dehydrogenase", bins=20)
  plt.hist(acdh_pHMM_acdh, label="acetaldehyde dehydrogenase", bins=20)
  plt.xlabel("Log Likelihood of acetaldehyde dehydrogenase Profile HMM on holdout genes", fontsize=20)
  plt.ylabel("Count", fontsize=20)
  plt.legend(prop={'size' : 24})
  plt.savefig("../output/figures/ACDH_pHMM.png")
  plt.show()

if __name__ == "__main__":
  main()