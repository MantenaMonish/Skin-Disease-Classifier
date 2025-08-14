# Skin Disease Classification using CNN and Transfer Learning

## 📖 Project Overview

This project explores and compares two deep learning approaches for classifying skin diseases from a self-collected dataset. It contains two separate Jupyter Notebooks: one implementing a **basic Convolutional Neural Network (CNN) from scratch**, and the other leveraging a state-of-the-art **Transfer Learning** model (MobileNetV2). The goal is to demonstrate the effectiveness and compare the performance of these two distinct methodologies.

## 📂 Project Structure

The repository is organized as follows. Note that the image data may be in nested subdirectories.

```
Skin-Disease-Classifier/
├── Dataset/
│   ├── acne/
│   │   └── acne/             # Image files are here
│   │       ├── image001.jpg
│   │       └── ...
│   ├── Nail_psoriasis/       # Or directly here
│   │   ├── image001.jpg
│   │   └── ...
│   └── ...                   # Other class folders
├── venv/                     # Virtual environment files
├── .gitignore                # Git ignore file
├── notes                     # Project notes
├── README.md                 # You are here!
├── requirements.txt          # Project dependencies
├── skin_disease_classification_cnn.ipynb       # Notebook 1: Basic CNN Model
└── skin_disease_classification_transfer.ipynb  # Notebook 2: Transfer Learning Model
```

## 🙏 Acknowledgements

This dataset is publicly available and was contributed by **sharun akter khushbu**. We extend our gratitude for their work in collecting and sharing this valuable data.

-   **Source:** Mendeley Data
-   **Dataset Title:** Skin Disease Classification Dataset
-   **Link:** [https://data.mendeley.com/datasets/3hckgznc67/1](https://data.mendeley.com/datasets/3hckgznc67/1)


## 📊 Results and Analysis

The primary experiment involved comparing the performance of the MobileNetV2 transfer learning model on the original dataset versus an enhanced (augmented) dataset. The results show the validation accuracy and loss over 15 epochs.

**Key Observations:**

-   **Accuracy:** The model trained on the **Original Dataset** consistently outperformed the model trained on the Enhanced Dataset. Its accuracy steadily climbed to over 70%.
-   **Loss:** The validation loss for the original data model decreased steadily, indicating effective learning. In contrast, the loss for the enhanced data model was erratic and significantly higher.

**Conclusion:**
For this specific architecture and dataset, the data augmentation strategy was counterproductive. This suggests that the MobileNetV2 model is already robust enough to handle the natural variations in the dataset, and the artificial transformations introduced too much noise, making it harder for the model to learn the underlying features of the diseases. The model trained on the original, clean data proved to be the superior approach.

---

## 🛠️ Technologies Used

-   **TensorFlow & Keras:** For building and training the deep learning models.
-   **NumPy:** For numerical operations and data manipulation.
-   **Matplotlib:** For data visualization and plotting results.
-   **Jupyter Notebook:** As the development environment for both experiments.

---

## 🚀 Setup and Installation

Follow these steps to set up the project environment.

**1. Clone the Repository**

```bash
git clone [https://github.com/MantenaMonish/Skin-Disease-Classifier.git](https://github.com/MantenaMonish/Skin-Disease-Classifier.git)
cd Skin-Disease-Classifier
```

**2. Create and Activate an Environment**
You can use either `venv` (standard Python) or `conda`.

-   **Option A: Using `venv`**
    ```bash
    # Create the virtual environment
    python -m venv venv
    # Activate it
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

-   **Option B: Using `conda`**
    ```bash
    # Create the conda environment
    conda create --name skindetect python=3.9
    # Activate it
    conda activate skindetect
    ```

**3. Install Dependencies**
Install the required libraries using the appropriate package manager.

-   **If using `venv` (pip):**
    ```bash
    pip install -r requirements.txt
    ```

-   **If using `conda`:**
    ```bash
    conda install --file requirements.txt
    ```


## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
