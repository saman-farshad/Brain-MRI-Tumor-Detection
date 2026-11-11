import argparse
from pathlib import Path
import joblib
from train import image_features

parser = argparse.ArgumentParser(description='Educational MRI image classifier')
parser.add_argument('--image', required=True, help='Path to an MRI image')
parser.add_argument('--model', default='models/brain_mri_model.joblib')
args = parser.parse_args()

if not Path(args.model).exists():
    raise FileNotFoundError('Train the model first: python train.py')

features = image_features(args.image)
if features is None:
    raise ValueError('Cannot read the image.')
model = joblib.load(args.model)  # Load only locally trained/trusted joblib files.
y_pred = model.predict([features])[0]
print('Prediction:', 'TUMOR' if y_pred == 1 else 'NO TUMOR')
print('Educational output only; not a medical diagnosis.')
