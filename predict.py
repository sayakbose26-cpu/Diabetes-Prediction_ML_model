
import pandas as pd
import joblib

model = joblib.load("diabetes_model.pk1")

print("\n====================================")
print("       DIABETES PREDICTION")
print("====================================")

age = int(input("Enter your age: "))
hypertension = int(input("Enter hypertension status (0/1): "))
heart_disease = int(input("Enter heart disease status (0/1): "))
smoking_history = input(
    "Enter smoking status (never/no info/current/former/ever): "
)
bmi = float(input("Enter your BMI: "))
HbA1c_level = float(input("Enter your HbA1c level: "))
blood_glucose_level = float(input("Enter your blood glucose level: "))

user_data = pd.DataFrame([{
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "smoking_history": smoking_history,
    "bmi": bmi,
    "HbA1c_level": HbA1c_level,
    "blood_glucose_level": blood_glucose_level
}])

prediction = model.predict(user_data)[0]
prob = model.predict_proba(user_data)[0]

classes = list(model.classes_)
confidence = prob[classes.index(prediction)] * 100

print("\n====================================")

if prediction == 1:
    print("RESULT: DIABETES DETECTED")
else:
    print("RESULT: NO DIABETES DETECTED")

print("Prediction probability: {:.2f}%".format(confidence))

print("====================================")
