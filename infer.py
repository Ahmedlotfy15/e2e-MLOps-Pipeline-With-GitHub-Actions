import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/Survival_NN.onnx")

input_name = session.get_inputs()[0].name

output_name = session.get_outputs()[0].name

def predict_survival(input_features):

    input_data = np.array(input_features, dtype=np.float32).reshape(1, -1)

    output = session.run([output_name], {input_name: input_data})[0]

    probability = 1 / (1 + np.exp(-output[0][0]))

    prediction = "Survived" if probability > 0.5 else "Not Survived"

    return prediction, probability


if __name__ == "__main__":
    input_features = [
        3,
        1,
        30,
        0,
        0,
        0,
    ]

    prediction, probability = predict_survival(input_features)

    print(f"Prediction: {prediction}")
    print(f"Probability: {probability}")





