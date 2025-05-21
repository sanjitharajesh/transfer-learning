# Transfer Learning for Lung Disease Detection using EfficientNet-B0

This project applies sequential transfer learning using EfficientNet-B0 to classify chest X-ray images for successful lung disease detection. 

## 📁 Dataset Sources

- **Pneumonia Dataset:** [Hugging Face](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia)
- **TB Dataset:** [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj)

## 🧰 Tools and Libraries

- Python 3.9+
- PyTorch
- torchvision
- scikit-learn
- albumentations
- matplotlib / seaborn
- Streamlit (optional for deployment)

## 🧪 Model Architecture

- **Base:** EfficientNet-B0 (pretrained on ImageNet)
- **Phase 1:** Fine-tuned on pneumonia dataset
- **Phase 2:** Sequentially fine-tuned on TB dataset
  - Unfreezes blocks 4–7
  - Includes class weighting, label smoothing, and threshold tuning
- **Alternative Approach Explored:** Domain-Adversarial Neural Network (DANN) for domain adaptation (explored)

## 📊 Results

| Model               | Accuracy | Normal Recall | TB Recall | AUC    |
|--------------------|----------|----------------|-----------|--------|
| Pneumonia Model    | 92%      | 97%            | 91%       | 0.98   |
| TB (DANN)          | 70%      | 81%            | 59%       | ~0.75  |
| TB (Final Model)   | 95%      | 100%           | 94%       | 0.98+  |



