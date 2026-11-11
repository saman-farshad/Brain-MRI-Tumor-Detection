import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Binary image classification: 0 = no tumor, 1 = tumor.
# Educational project only. Not intended for medical diagnosis.
DATA_PATH = Path('data')
IMAGE_SIZE = 128


def image_features(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    features = hog(image, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), feature_vector=True)
    return features


def read_data(data_path):
    x, y, image_names = [], [], []
    folders = [('no', 0), ('yes', 1)]

    for folder_name, label in folders:
        folder_path = data_path / folder_name
        if not folder_path.exists():
            raise FileNotFoundError(f'Missing folder: {folder_path}')

        for image_path in sorted(folder_path.rglob('*')):
            if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            features = image_features(image_path)
            if features is not None:
                x.append(features)
                y.append(label)
                image_names.append(str(image_path))

    if len(x) == 0 or len(set(y)) != 2:
        raise ValueError('Add labeled MRI images to data/no and data/yes.')
    return np.array(x), np.array(y), image_names


def metrics(y_train, y_test, y_pred_train, y_pred_test):
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, pos_label=1, zero_division=0)

    conf_matrix = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    print('-' * 45)
    print(f'Accuracy (Train) : {acc_train*100:.2f}%')
    print(f'Accuracy (Test)  : {acc_test*100:.2f}%')
    print(f'Precision        : {precision*100:.2f}%')
    print(f'Recall           : {recall*100:.2f}%')
    print(f'F1 Score         : {f1*100:.2f}%')
    print('-' * 45)

    print('Confusion Matrix:')
    print(pd.DataFrame(conf_matrix,
                       index=['Actual No Tumor', 'Actual Tumor'],
                       columns=['Pred No Tumor', 'Pred Tumor']))

    os.makedirs('outputs', exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(conf_matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(conf_matrix[i, j]), ha='center', va='center')
    ax.set_xticks([0, 1], ['No Tumor', 'Tumor'])
    ax.set_yticks([0, 1], ['No Tumor', 'Tumor'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Brain MRI - Confusion Matrix')
    fig.tight_layout()
    fig.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.close(fig)
    return acc_test, precision, recall, f1


if __name__ == '__main__':
    x, y, image_names = read_data(DATA_PATH)
    df = pd.DataFrame({'image': image_names, 'label': y})
    print('Dataset size:', len(df))
    print(df['label'].value_counts().rename(index={0: 'No Tumor', 1: 'Tumor'}))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )

    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', SVC(class_weight='balanced'))
    ])

    p_g = {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=p_g,
        scoring='f1',
        cv=3,
        n_jobs=-1,
        error_score='raise'
    )
    grid.fit(x_train, y_train)

    print('Best Parameters:', grid.best_params_)
    print('Best CV F1 Score:', round(grid.best_score_, 4))
    best_model = grid.best_estimator_
    y_pred_train = best_model.predict(x_train)
    y_pred_test = best_model.predict(x_test)
    acc, precision, recall, f1 = metrics(y_train, y_test, y_pred_train, y_pred_test)

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/brain_mri_model.joblib')
    pd.DataFrame([{'accuracy': acc, 'precision': precision,
                   'recall': recall, 'f1': f1}]).to_csv('outputs/metrics.csv', index=False)
    print('Saved: models/brain_mri_model.joblib')
