# Diabetes Prediction ML Project
### Overview
This project aims to predict whether a person is diabetic or not based on various health metrics. The dataset used for this project is the Pima Indian Diabetes dataset, which includes features like the number of pregnancies, glucose levels, blood pressure, and more.

### Table of Contents

1. Installation
2. Dataset
3. Data Preprocessing
4. Model Training
5. Evaluation
6. Prediction
7. Usage


### Installation
To run this project, you need Python installed on your system along with the following libraries:

pip install numpy pandas scikit-learn

Alternatively, you can open the project in Google Colab and run the cells directly.

### Dataset
The dataset used is the Pima Indian Diabetes dataset. It contains the following columns:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0: Non-diabetic, 1: Diabetic)
  
The dataset can be loaded into a pandas DataFrame as shown below:

#### diabetes_dataset = pd.read_csv("/content/diabetes.csv")

### Data Preprocessing
#### 1. Loading the dataset:

diabetes_dataset = pd.read_csv("/content/diabetes.csv")

#### 2. Inspecting the dataset:

diabetes_dataset.head()
diabetes_dataset.shape
diabetes_dataset.describe()
diabetes_dataset["Outcome"].value_counts()

#### 3. Separating features and labels:

X = diabetes_dataset.drop(columns = "Outcome", axis=1)
Y = diabetes_dataset["Outcome"]

#### 4. Standardizing the data:

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data

### Model Training
#### 1. Splitting the data:

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#### 2. Training the SVM model:

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

### Evaluation

#### 1. Accuracy on training data:

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data:', training_data_accuracy)

#### 2. Accuracy on test data:

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data:', test_data_accuracy)

### Prediction
To make a prediction for a new data point:

#### 1. Input data:

input_data = (1, 163, 72, 0, 0, 39, 1.222, 33)
input_data_nmp = np.asarray(input_data)
input_data_reshape = input_data_nmp.reshape(1, -1)

#### 2. Standardize and predict:

std_data = scaler.transform(input_data_reshape)
prediction = classifier.predict(std_data)

#### 3. Output the result:

if prediction[0] == 0:
    print("The Person is not diabetic")
else:
    print("The Person is diabetic")
    
### Usage

1. Clone the repository or download the script.
2. Install the required libraries.
3. Run the script in a Jupyter notebook or any Python environment.
