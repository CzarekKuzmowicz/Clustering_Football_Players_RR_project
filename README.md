# Clustering of Top 5 European League Football Players

Project prepared for the Reproducible Research 2026 course.

## Group members

- Cezary Kuźmowicz
- Filip Żebrowski
- Dariusz Doktorski

## Research question

Can player performance statistics from the 2021/22 season be used to identify meaningful groups of footballers across Europe's top five leagues? We want to examine whether unsupervised clustering reveals distinct player profiles based on attacking, passing, defensive, and aerial metrics.

## Data source

The project uses the public Kaggle dataset ["2021-2022 Football Player Stats"](https://www.kaggle.com/datasets/vivovinco/20212022-football-player-stats). The dataset was prepared by the Kaggle author based on data collected from [FBref](https://fbref.com/en/).

## Planned approach

We plan to clean and filter the data, remove players with very limited playing time, and select a smaller set of informative variables for clustering. The main analytical method is k-means clustering, supported by cluster tendency checks, dimensionality reduction with PCA, and visualizations that help interpret the resulting player groups.

## Language / Tools

The project is implemented in Python. We currently use `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and Jupyter Notebook for data preparation, clustering, evaluation, and visualization.

## Running notebooks with Docker

The Docker image uses Python 3.11.8 and installs all Python dependencies needed by both notebooks:

- `Season 21-22 (reproduction).ipynb`
- `Season 22-23 (extraction).ipynb`

Build and start JupyterLab from the project root:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8888
```

Stop JupyterLab with `Ctrl+C`.

The project directory is mounted into the container, so notebook edits and generated files are saved on the host machine.

## Motivation

Football data combines domain knowledge with rich, high-dimensional statistics, which makes it a strong fit for exploratory clustering. Our group finds this interesting because it connects machine learning methods with a sport we actively follow and gives us a way to compare player styles beyond standard position labels.

## Original project

We are reproducing same project conducted in R:
https://rpubs.com/czarek_kuzmowicz/1262334

## Repository structure

```text
├── data/                                  # Raw player statistics (CSV)
├── notebooks/
│   ├── Season 21-22 (reproduction).ipynb  # Core reproduction of the R project
│   └── Season 22-23 (extraction).ipynb    # Extension to new data
├── Dockerfile                             # Docker configuration
├── docker-compose.yml                     # Docker Compose setup
├── requirements.txt                       # Pinned Python dependencies
└── README.md
```

## Alternative Setup (Local Environment)
### If you don't want to use Docker:

Create a virtual environment: python -m venv .venv

Activate it: source .venv/bin/activate (or .venv\Scripts\activate on Windows)

Install dependencies: pip install -r requirements.txt

Run Jupyter: jupyter lab

## Reproducibility Features
Relative Paths: No absolute paths used (e.g., /Users/name/...). All data is accessed via relative directory structures.

Fixed Random Seeds: To ensure identical clustering results across different machines, random_state is fixed in all K-Means and PCA functions.

Pinned Versions: All library versions in requirements.txt are pinned to prevent breaking changes.

## How to run

Start JupyterLab (via Docker or local setup).

Open Season 21-22 (reproduction).ipynb and run all cells to see the reproduction of the original R study.

Open Season 22-23 (extraction).ipynb to see how the model performs on the latest data.

Execution time: Approximately 2-3 minutes on a standard machine.