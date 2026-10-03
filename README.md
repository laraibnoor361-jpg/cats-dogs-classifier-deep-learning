# 🐾 PawVision AI: Cats vs Dogs Classifier

A deep learning image classifier built with **TensorFlow/Keras** (MobileNetV2 transfer learning) and a premium dark-themed **Streamlit** dashboard.

Dataset: images taken from the [Cats and Dogs Mini Dataset (Kaggle)](https://www.kaggle.com/datasets/aleemaparakatta/cats-and-dogs-mini-dataset). A 32-image subset (18 cats, 14 dogs) is included in the `dataset/` folder, so the project is fully self-contained.

## ✨ Features

- Drag-and-drop uploader for `.jpg` / `.png` images
- Side-by-side image and prediction view
- Cat vs Dog confidence bars and metric cards
- Model Insights tab: accuracy, precision, recall, F1, training curves, confusion matrix
- Adjustable confidence threshold and sample image picker in the sidebar

## 📁 Project Structure

```
cats-dogs-classifier/
├── .streamlit/config.toml      # Theme and brand colors
├── dataset/
│   ├── cats/                   # cat images
│   └── dogs/                   # dog images
├── models/                     # Trained model + metrics (created by train.py)
├── sample_images/              # Demo images for the app sidebar
├── app.py                      # Streamlit web app
├── train.py                    # Training script
├── make_samples.py             # Copies demo images from dataset/ to sample_images/
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧠 How the Model Works

**Architecture:** Rescaling → MobileNetV2 (pre-trained on ImageNet) → Global Average Pooling → Dropout (0.3) → Dense(1, sigmoid)

1. **Input**: images are resized to 224×224 RGB.
2. **Augmentation**: random flip, rotation, zoom and contrast (training only). With a small dataset, this shows the network many slightly different versions of each photo and reduces overfitting.
3. **Rescaling**: pixels are mapped to [-1, 1], the range MobileNetV2 expects. This layer is part of the saved model, so the app needs no extra preprocessing.
4. **Feature extraction**: MobileNetV2's depthwise-separable convolutions turn the image into a 1280-number feature vector. It is fast and lightweight, which suits free cloud hosting.
5. **Classification head**: a single sigmoid neuron outputs P(dog). P(cat) = 1 − P(dog).
6. **Training in two phases**:
   - *Phase 1*: the MobileNetV2 base is frozen and only the new head is trained (learning rate 1e-3).
   - *Phase 2*: the top 30 layers of the base are unfrozen and fine-tuned with a very small learning rate (1e-5).
7. **Data split**: about 70% train / 15% validation / 15% test (stratified, so both classes appear in every split). Early stopping and learning-rate reduction are used.

> **About the dataset size:** this is a *mini* dataset, so the test set is only a handful of images. Accuracy numbers on so few images can swing a lot from run to run (one image changes the score by 20%). Transfer learning from ImageNet is what makes it work well even with little data. To get steadier results, add more images to `dataset/cats` and `dataset/dogs` and train again.

## 🚀 Run Locally

```bash
# 1. Create environment (Python 3.11 recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train (uses the images in dataset/, needs internet once to fetch ImageNet weights)
python train.py
#    -> creates models/cats_dogs_model.h5, .keras, metrics.json, class_names.json, charts
#    -> refreshes sample_images/ with unseen test images

# 4. Launch the app
streamlit run app.py
```

Any dataset layout works: an image is used if its folder or file name contains "cat" or "dog". To use the full Kaggle dataset instead, run `pip install kagglehub` and then `python train.py --download`.

### No TensorFlow on your PC? Train on Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com) → New notebook.
2. Upload the project zip (left sidebar → Files), then run:
   ```python
   !unzip -q cats-dogs-classifier.zip
   %cd cats-dogs-classifier
   !pip install -q scikit-learn matplotlib pillow
   !python train.py
   ```
3. Download the `models/` folder (zip it with `!zip -r models.zip models sample_images`) and copy its contents into your local project folder.

## ☁️ Deploy on Streamlit Community Cloud

1. **Train first** so that `models/cats_dogs_model.h5` and `models/metrics.json` exist. The `.gitignore` is set up to keep the final model files and charts while ignoring other junk.
2. **Push to GitHub** (or upload the files through the GitHub website):
   ```bash
   git init
   git add .
   git commit -m "Cats vs Dogs classifier"
   git branch -M main
   git remote add origin https://github.com/<your-username>/cats-dogs-classifier.git
   git push -u origin main
   ```
   Check on GitHub that the `models/` folder contains the `.h5` model and `metrics.json`, and that `dataset/` and `sample_images/` have images.
3. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
4. Click **Create app** → **Deploy a public app from GitHub**.
5. Select your repository, branch `main`, and main file path `app.py`.
6. Open **Advanced settings** and choose **Python 3.11** (TensorFlow needs a supported version).
7. Click **Deploy**. The first build takes a few minutes while TensorFlow installs.

## ⚡ Deploy without training first (auto-training)

If `models/` has no trained model when the app starts, `app.py` trains one automatically from the `dataset/` folder (MobileNetV2 head only, about 2-5 minutes on the first visit) and then shows the normal dashboard. This lets you deploy straight from GitHub without running Python on your own PC.

Keep in mind: Streamlit Cloud storage is temporary, so after the app restarts or sleeps for a long time it will train again. For a faster and more stable demo, train once (locally or on Colab) and commit the `models/` files, as described above.

## 🛠 Troubleshooting

- **"Model file not found"**: run `train.py` and make sure the `models/` files were committed to GitHub.
- **Build fails on TensorFlow**: set Python to 3.11 in Advanced settings and reboot the app.
- **Low or unstable accuracy**: add more images to the dataset, or increase `--epochs` and `--fine_tune_epochs`.

## 🎤 Quick Viva Notes

- **What is transfer learning?** Re-using a network already trained on millions of images (ImageNet) and only training a small new top layer for our two classes.
- **Why MobileNetV2?** It is small and fast, so it trains on a CPU and runs on free cloud hosting.
- **What does the sigmoid output mean?** A number between 0 and 1: the probability that the image is a dog. Cat probability is 1 minus that.
- **Why augmentation?** The dataset is small. Random flips/rotations/zooms create variety and stop the model from memorising the training photos.
- **Why a validation set and a test set?** Validation guides training (early stopping); the test set is never used in training and gives the honest final accuracy.
- **Why freeze the base first, then fine-tune?** Training the new head first avoids damaging the pre-trained features; fine-tuning afterwards with a tiny learning rate adapts them gently.

## 📜 License

MIT. Free to use for learning and projects.
