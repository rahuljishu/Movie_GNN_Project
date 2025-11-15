# Movie Recommendation System
## Structural Network Analysis and Graph ML
---
## 🌟 Core Project Idea

This project develops an **advanced movie recommendation system** by modeling the relationships between actors and movies as a complex graph network. By moving beyond traditional methods that treat movies as isolated items, this system leverages the deep relational structure of the film industry to provide recommendations that are **accurate, personalized, and contextually aware**.

The project is a **comprehensive, end-to-end demonstration** of a graph-based machine learning pipeline, covering:

- **Data Acquisition and Graph Construction** from raw industry data
- **Classical Structural Analysis** using concepts like degree distribution, shortest path algorithms, and clique detection to uncover insights about the network
- **GNN Model Training** using PyTorch Geometric to build and train a HeteroGNN that learns from the graph's structure
- **Final Model Evaluation and Recommendation** to prove the model's effectiveness and provide a live demonstration

---

## 🚀 Key Findings

### Structural Insights
The collaboration network was found to follow a **long-tail distribution**, with a small elite of hyper-connected actors. A **maximal clique of 52 members** was discovered, revealing a massive, densely connected "creative core" within the industry.

### Model Performance
The final Graph Neural Network achieved a **Test RMSE of 1.0753**, demonstrating a high degree of accuracy in predicting user ratings on a 5-star scale.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9 |
| **Core Libraries** | Pandas, NetworkX, Matplotlib, Seaborn |
| **Machine Learning** | PyTorch, PyTorch Geometric (PyG) |
| **Environment** | Conda |

---

## 📋 Prerequisites

- **Anaconda** or **Miniconda** installed on your system
- **CUDA-compatible NVIDIA GPU** (recommended for GNN training)
- Stable internet connection for downloading datasets

---

## 🔧 Reproducing the Project

Follow these steps to set up the environment and run the project from start to finish.

### Step 1: Clone the Repository

```bash
git clone https://github.com/rahuljishu/Movie_GNN_Project.git
cd Movie_GNN_Project
```

### Step 2: Set Up the Conda Environment

#### Create the Conda Environment

```bash
conda create --name graph_project python=3.9
```

#### Activate the Environment

```bash
conda activate graph_project
```

#### Install All Required Libraries

The `requirements.txt` file contains all necessary packages.

```bash
pip install -r requirements.txt
```

### Step 3: Download the Raw Data

⚠️ **Important**: This project requires external data that is not included in the repository due to its size.

#### Create the Directory

Make sure you have a `data/raw` folder inside the main project directory:

```bash
mkdir -p data/raw
```

#### Download IMDb Datasets

1. Visit the [IMDb Non-Commercial Datasets page](https://datasets.imdbws.com/)
2. Download the following files and save them directly into your `data/raw/` folder (**do not unzip them**):
   - `name.basics.tsv.gz`
   - `title.basics.tsv.gz`
   - `title.principals.tsv.gz`

#### Download MovieLens Dataset

1. Visit the [MovieLens Latest Datasets page](https://grouplens.org/datasets/movielens/latest/)
2. Download `ml-latest.zip`
3. Unzip the file (this will create a folder named `ml-latest`)
4. Move the entire `ml-latest` folder into your `data/raw/` directory

#### Verify Your Data Structure

After completing this step, your `data/raw/` folder should look like this:

```
data/raw/
├── ml-latest/
│   ├── links.csv
│   ├── movies.csv
│   ├── ratings.csv
│   └── ... (other files)
├── name.basics.tsv.gz
├── title.basics.tsv.gz
└── title.principals.tsv.gz
```

### Step 4: Run the Jupyter Notebooks in Order

The project is divided into **four notebooks** that must be run **sequentially**.

#### Start Jupyter

From your Anaconda Prompt (with the `graph_project` environment active), run:

```bash
jupyter notebook
```

#### Run the Notebooks

Execute the following notebooks in order:

1. **`notebooks/1_data_preparation_and_graph_build.ipynb`**
   - Processes the raw data and creates the main graph file
   - ⚠️ **This must be run first**

2. **`notebooks/2_classical_structural_analysis.ipynb`**
   - Performs the classical graph theory analysis
   - Explores degree distributions, shortest paths, and clique detection

3. **`notebooks/3_gnn_model_training.ipynb`**
   - Builds and trains the GNN
   - ⚠️ **Note**: This step requires a CUDA-compatible NVIDIA GPU for reasonable performance

4. **`notebooks/4_model_evaluation_and_visualization.ipynb`**
   - Loads the trained model from the `models/` folder
   - Evaluates model performance
   - Provides a live recommendation demo

---

## 📊 Project Structure

```
Movie_GNN_Project/
├── data/
│   ├── raw/                    # Raw datasets (user must download)
│   └── processed/              # Generated during notebook execution
├── notebooks/
│   ├── 1_data_preparation_and_graph_build.ipynb
│   ├── 2_classical_structural_analysis.ipynb
│   ├── 3_gnn_model_training.ipynb
│   └── 4_model_evaluation_and_visualization.ipynb
├── models/                     # Trained models saved here
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🎯 Key Features

- **Graph-Based Architecture**: Models movies and actors as interconnected nodes
- **Heterogeneous GNN**: Captures different relationship types in the network
- **Classical Analysis**: Provides insights through traditional graph theory metrics
- **End-to-End Pipeline**: Complete workflow from data acquisition to deployment
- **Live Recommendations**: Interactive demo for testing the system

---

## 📈 Results

- **RMSE**: 1.0753 on 5-star rating scale
- **Network Insights**: Discovered 52-member maximal clique
- **Distribution**: Long-tail collaboration network with elite core

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **IMDb** for providing comprehensive film industry data
- **GroupLens** for the MovieLens dataset
- **PyTorch Geometric** team for the excellent GNN library

---

## 📧 Contact
  
GitHub: [@rahuljishu](https://github.com/rahuljishu)

---

## ⚠️ Troubleshooting

### Common Issues

**Issue**: CUDA out of memory during training
- **Solution**: Reduce batch size in notebook 3 or use a GPU with more memory

**Issue**: Missing data files
- **Solution**: Ensure all files are downloaded to `data/raw/` as specified in Step 3

**Issue**: Import errors
- **Solution**: Verify that the `graph_project` environment is activated and all dependencies are installed

---

**Happy Coding! 🚀**