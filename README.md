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

## Motivation

Football data combines domain knowledge with rich, high-dimensional statistics, which makes it a strong fit for exploratory clustering. Our group finds this interesting because it connects machine learning methods with a sport we actively follow and gives us a way to compare player styles beyond standard position labels.
