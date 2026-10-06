# PetFinder Data Visualization Setup Guide

This guide walks you through setting up the Kaggle API, downloading the PetFinder dataset, and running our visualization notebook in Google Colab.

## 1. Get Your Kaggle API Key
To download the data directly into Colab, you need a personal API key.
1. Log in to [Kaggle.com](https://www.kaggle.com/).
2. Navigate to your **Settings** (click your profile picture in the top right > Settings).
3. Scroll down to the **API** section and click **Create New Token**.
4. A file named `kaggle.json` will download to your computer. Keep this handy.

## 2. Accept Competition Rules (Important!)
Before the API will let you download the data, you *must* accept the competition rules. If you skip this step, the notebook will throw a "403 Forbidden" error.
1. Go to the [PetFinder Competition Rules Page](https://www.kaggle.com/c/petfinder-adoption-prediction/rules).
2. Scroll down and click the **"I Understand and Accept"** button.

## 3. Set Up Google Colab
1. Open [Google Colab](https://colab.research.google.com/) and open our shared notebook.
2. Run the very first setup cell in the notebook (the one containing the `files.upload()` command). 
3. A "Choose Files" button will appear in the output area. Click it and select the `kaggle.json` file you downloaded in Step 1.

## 4. Run the Notebook
1. Once your API key is uploaded, the notebook will automatically authenticate, download, and unzip the PetFinder dataset.
2. You can now execute the rest of the notebook by clicking the "Play" button on each cell sequentially, or by going to the top menu and clicking **Runtime > Run all**.
3. Scroll down to view the loaded data and the generated Seaborn visualizations!