from rdkit.Chem import MolFromSmiles, Draw
import sys
sys.path.append('c:/Users/ricar/Documents/GitHub/huss-pred-webtool')
from huss_pred.huss_pred import main

# to set the coloring display of the outcomes on the website
COLORS = {
    "Sensitizer": "red",
    "Non-sensitizer": "green",
}


AD_MEANING = {
    "Inside": "The molecule is within the applicability domain of the model",
    "Outside": "The molecule is outside the applicability domain of the model",
    "": ""
}

color_text = False  # set to True if you want to color code the text


def get_molecule_data_from_smiles(smiles_str, options):
    molecule = MolFromSmiles(smiles_str)

    if molecule is None:
        return None

    data = main(smiles_str, **options)

    # add color coding of text
    if color_text:
        data = [_ + [COLORS[_[1]]] for _ in data]
    else:
        data = [_ + ["black"] for _ in data]

    print(options)

    # if "make_prop_img" in options.keys():
    #     print(data)
    #     data = [_ + [AD_MEANING[_[1]]] for _ in data]
    # else:
    data = [_ + ["Red contributes to toxicity, Green contributes to non-toxicity"] for _ in data]

    # skip if no models selected
    if len(data) == 0:
        return None

    data_to_return = {
        'svg': Draw.MolsToGridImage([molecule], useSVG=True, molsPerRow=1),
        'SMILES': smiles_str,
        'pred_data': data,
    }

    print(data_to_return)

    return data_to_return
