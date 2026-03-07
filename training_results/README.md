# Training Results

Place training output images in the model-specific folders below.  
The Streamlit app will automatically detect and display them under **Training Results**.

## Folder Structure

```
training_results/
├── yolo/
│   ├── results.png            # Training metrics over epochs
│   ├── confusion_matrix.png   # Confusion matrix
│   ├── PR_curve.png           # Precision-Recall curve (optional)
│   └── F1_curve.png           # F1 curve (optional)
├── rtdetr/
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── PR_curve.png
│   └── F1_curve.png
└── frcnn/
    ├── results.png
    ├── confusion_matrix.png
    ├── PR_curve.png
    └── F1_curve.png
```

## How to add images

After training in Google Colab, download the plots from the training run directory
and place them in the appropriate folder above.

Supported image formats: `.png`, `.jpg`, `.jpeg`
