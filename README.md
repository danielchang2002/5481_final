# 5481 Final: Protein Family Classification Using Profile HMMs 

Daniel Chang, Levi Cavagnetto, and Garrett Abou-Zeid

Using a toy dataset, this repo demonstrates a powerful algorithm for protein family classification.
We build profile HMMs for two genes: alcohol dehydrogenase (ADH) and acetaldehyde dehydrogenase (ACDH). 

# Requirements
- NumPy
- Matplotlib
- tqdm (optional)

## How to Run

To train the HMMs on the two genes, please run:
```bash
cd src
python3 hmm.py 
```

To plot the scores and analyze the performance of the models, please run:
```bash
python3 plot.py
```


