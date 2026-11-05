# AI Coursework Projects

A collection of university projects exploring classic AI and machine learning algorithms — from evolutionary optimization to neural networks implemented from scratch. Each project includes its own documentation, source code, and result visualizations.

## Projects

| # | Project | Algorithms | Description |
|---|---------|-----------|-------------|
| 01 | [Traveling Salesman Problem](./01-travelling-salesman-problem) | Genetic Algorithm, Tabu Search, Simulated Annealing | Finding near-optimal routes through 20–40 randomly placed cities |
| 02 | [Zen Garden](./02-zen-garden) | Evolutionary Algorithm | Simulating a monk raking sand into straight rows around obstacles |
| 03 | [Clustering (k-means & divisive)](./03-clustering-kmeans) | k-means (centroid), k-means (medoid), Divisive clustering | Clustering ~40,000 2D points into groups with bounded intra-cluster spread |
| 04 | [Clustering (agglomerative)](./04-clustering-hierarchical) | Agglomerative clustering (centroid & medoid) | Bottom-up hierarchical clustering with an optimized distance matrix |
| 05 | [Crypto Price Classification](./05-crypto-classification) | LSTM / GRU / 1D-CNN / Transformer | Predicting next-day price direction from historical time series |
| 06 | [Backpropagation from Scratch](./06-backpropagation) | Custom NumPy implementation (no autograd) | Modular feedforward network trained on AND / OR / XOR |

## Tech Stack

- Python 3
- NumPy, Matplotlib
- PyTorch (project 05 only)
- No autograd libraries used in project 06 — backpropagation is implemented manually

## Repository Structure

```
ai-coursework-projects/
├── README.md
├── 01-traveling-salesman/
│   ├── README.md
│   ├── src/
│   ├── results/
│   └── requirements.txt
├── 02-zen-garden/
├── 03-clustering-kmeans-divisive/
├── 04-clustering-agglomerative/
├── 05-crypto-classification/
└── 06-backpropagation/
```

Each project folder is self-contained and includes:

- `README.md` — problem description, algorithm details, and results
- `src/` — source code
- `results/` — plots, visualizations, and sample outputs
- `requirements.txt` — dependencies for that specific project

## How to Run

Each project can be run independently. Navigate to the project folder and install its dependencies:

```bash
cd 01-traveling-salesman
pip install -r requirements.txt
python main.py
```

See each project's own README for specific usage instructions and parameters.
