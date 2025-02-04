#
import base64
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import SimilarityMaps
from scipy.spatial.distance import cdist
import numpy as np
from scipy.spatial.distance import pdist, cdist

"""import glob
import gzip
import bz2"""
import os
#import _pickle as cPickle

import io
import matplotlib.pyplot as plt

# god hates me so in my version of python I cannot supress these damn user warning so I do this nuclear option instead
"""import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn"""

# for setting confidence-based AD:
AD_THRESH = 0.6


import joblib  # Ensure this import is at the beginning of your script


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")  # Directory where models are stored

MODEL_DICT = {
    'Binary Sensitization': [joblib.load(os.path.join(MODEL_DIR, 'ss_DSA05_binary_morgan_r2_2048_svm_calibrated_with_threshold.joblib'))['model']],
    'Multiclass Sensitization Potency': [joblib.load(os.path.join(MODEL_DIR, 'DSA05_mordred_rf_multiclass.joblib'))]
}

THRES_DICT = {
    'Binary Sensitization': [joblib.load(os.path.join(MODEL_DIR, 'ss_DSA05_binary_morgan_r2_2048_svm_calibrated_with_threshold.joblib'))['threshold']],
}

# CHECK THIS
AD_DICT = {
    'Binary Sensitization': [joblib.load(os.path.join(MODEL_DIR, 'binary_AD.pkl'))],
    'Multiclass Sensitization Potency': [joblib.load(os.path.join(MODEL_DIR, 'multiclass_AD.pkl'))],
}

FEATURES_DICT = {
    'Binary Sensitization': [np.load(os.path.join(MODEL_DIR, 'selected_features_svm_binary.npy'), allow_pickle=True)],
    'Multiclass Sensitization Potency': [np.load(os.path.join(MODEL_DIR, 'selected_features_rf_multiclass.npy'), allow_pickle=True)],
}

# lol I'm just like screw code readability sorry
MODEL_DICT_INVERT = {str(v): key for key, val in MODEL_DICT.items() for v in val}

CLASSIFICATION_DICT = {
    'Binary Sensitization Call': {
        0: "Non-sensitizer",
        1: "Sensitizer"
    },
    'Multiclass Sensitization Potency': {
        0: "Non-sensitizer",
        1: "Weak sensitizer",
        2: "Strong sensitizer",
    }
}

AD_DICT_BOOL = {
    True: "Inside",
    False: "Outside"
}


def _get_AD_thresh(training_smiles, file_name):
    fps = np.array([list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=2, nBits=2048, useFeatures=False))
                    for smi in training_smiles])

    dists = pdist(fps)
    mean_1 = dists.mean()
    dists_2 = dists[np.where(dists < mean_1)]
    mean_2 = dists_2.mean()
    std_2 = dists_2.std()

    threshold = mean_2 + (0.5 * std_2)

    import pickle
    pickle.dump((threshold, fps),  open(file_name, "wb"))

    with open(file_name, "wb") as f:
        pickle.dump((threshold, fps), f)


def calc_ad(query_fp, ad_tuple):
    dist = cdist(query_fp.reshape(1, -1), ad_tuple[1], "euclidean")
    return (dist < ad_tuple[0]).any()



def run_prediction_binary(model, smi, calculate_ad=True, ad_tup=None, threshold=0.5):
    """_summary_

    Args:
        model (_type_): _description_
        model_data (_type_): _description_
        smi (_type_): _description_
        calculate_ad (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    fp = np.zeros((2048, 1))
    # sub in your FP function
    _fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=2, nBits=2048, useFeatures=False)
    DataStructs.ConvertToNumpyArray(_fp, fp)

    selected_features = np.load(os.path.join(MODEL_DIR, 'selected_features_svm_binary.npy'))
    fp_selected = fp[selected_features]

    pred_proba = model.predict_proba(fp_selected.reshape(1, -1))[:, 1]
    pred = 1 if pred_proba > threshold else 0

    if pred == 0:
        pred_proba = 1 - float(pred_proba)

    # used to get proba of the inactive class if deemed inactive
    # if pred == 0:
    #     pred_proba = 1-pred_proba

    if calculate_ad:
        ad = calc_ad(fp, ad_tup)
        return pred, pred_proba, ad
    
    else:
        ad = False  # Set to True or False depending on how you want to interpret "no AD calculation"

    return pred, float(pred_proba), ""

from sklearn.preprocessing import MinMaxScaler
from mordred import Calculator, descriptors

def run_prediction_multiclass(model, smi, calculate_ad=True, ad_tup=None, threshold=0.5):
    """_summary_

    Args:
        model (_type_): _description_
        model_data (_type_): _description_
        smi (_type_): _description_
        calculate_ad (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    fp_ad_arr = np.zeros((2048, 1))
    # sub in your FP function
    fp_ad = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=2, nBits=2048, useFeatures=False)
    DataStructs.ConvertToNumpyArray(fp_ad, fp_ad_arr)

    calc = Calculator(descriptors, ignore_3D=True)
    mol = Chem.MolFromSmiles(smi)  # Convert SMILES to RDKit Mol object
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smi}")
    
    _fp = calc.pandas([mol])  # Pass the mol as a list

    # Min-Max scaling
    scaler = MinMaxScaler()
    fp = scaler.fit_transform(_fp)

    # Load selected features with allow_pickle=True
    selected_features = np.load(os.path.join(MODEL_DIR, 'selected_features_rf_multiclass.npy'), allow_pickle=True)

    # Ensure selected_features are valid column names
    # Filter the fp DataFrame to retain only the selected columns (features)
    fp_selected = _fp[selected_features]

    # Scale the selected features
    fp_selected_scaled = scaler.fit_transform(fp_selected)

    # Get probabilities for all classes (not just class 1)
    pred_proba = model.predict_proba(fp_selected_scaled.reshape(1, -1))

    # Get the class with the highest probability
    pred = np.argmax(pred_proba)

    # Probability of the predicted class
    predicted_class_proba = pred_proba[0][pred]

    if calculate_ad:
        ad = calc_ad(fp_ad_arr, ad_tup)
        return pred, predicted_class_proba, ad
    
    return pred, predicted_class_proba, ""





def get_prob_map(model, smi):
    def get_fp(mol, idx):
        fps = np.zeros((2048, 1))
        _fps = SimilarityMaps.GetMorganFingerprint(mol, idx, radius=2, nBits=2048)
        DataStructs.ConvertToNumpyArray(_fps, fps)
        selected_features = np.load(os.path.join(MODEL_DIR, 'selected_features_svm_binary.npy'))
        fp_selected = fps[selected_features]
        return fp_selected

    def get_proba(fps):
        return float(model.predict_proba(fps.reshape(1, -1))[:, 1])
    mol = Chem.MolFromSmiles(smi)
    d2d = Draw.MolDraw2DCairo(500, 500)
    SimilarityMaps.GetSimilarityMapForModel(mol, get_fp, get_proba, draw2d=d2d)
    fig = base64.b64encode(d2d.GetDrawingText()).decode("ascii")
    return fig

    # mol = Chem.MolFromSmiles(smi)
    # fig, _ = SimilarityMaps.GetSimilarityMapForModel(mol, get_fp, get_proba)
    # imgdata = io.StringIO()
    # fig.savefig(imgdata, format='svg')
    # imgdata.seek(0)  # rewind the data
    # plt.savefig(imgdata, format="svg", bbox_inches="tight")

    # return imgdata.getvalue()


def main(smi, calculate_ad=True, make_prop_img=False, **kwargs):
    values = {}

    for key, val in kwargs.items():
        if key in MODEL_DICT.keys():  # check if this kwarg is for a model
            if val:  # check if model is turned on
                model = MODEL_DICT[key][0]  # Get the model
                ad_tup = AD_DICT[key][0]

                # Choose the appropriate prediction function
                if key == 'Binary Sensitization':
                    pred, pred_proba, ad = run_prediction_binary(model, smi, calculate_ad=calculate_ad, ad_tup=ad_tup)
                elif key == 'Multiclass Sensitization Potency':
                    pred, pred_proba, ad = run_prediction_multiclass(model, smi, calculate_ad=calculate_ad, ad_tup=ad_tup)

                contrib_svg_str = ""
                if make_prop_img:
                    contrib_svg_str = get_prob_map(model, smi)

                values[key] = [pred, float(pred_proba), AD_DICT_BOOL[ad], contrib_svg_str]

    processed_results = []
    for key, val in values.items():
        # Use 'Binary Sensitization Call' for the classification dict key
        classification_key = 'Binary Sensitization Call' if key == 'Binary Sensitization' else key
        processed_results.append([key, CLASSIFICATION_DICT[classification_key][val[0]], val[1], val[2], val[3]])

    return processed_results






