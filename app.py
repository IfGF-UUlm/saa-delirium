import json
import numpy as np
from preprocessing.preprocessing import prepare_sample
from pod_predictor.inference import PODPredictor


if __name__ == "__main__":

    # Load and preprocess single sample
    with open("single_sample.json") as f:
        sample = json.load(f)
    sample = prepare_sample(sample)

    # Calculate probability
    pod_predictor = PODPredictor()
    delir_proba = {'delir_proba': pod_predictor.predict_proba(sample)[0, 0].item()}

    # Export to JSON
    with open('delir_proba.json', 'w') as f:
        json.dump(delir_proba, f, indent=4)
