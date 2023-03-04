#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <sstream>

using namespace std;

// Sigmoid function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Logistic regression function
void logisticRegression(vector<vector<double>> &X, vector<double> &y, vector<double> &theta, int maxIterations, double alpha) {
    int m = X.size();
    int n = X[0].size();
    vector<double> h(m);
    vector<double> grad(n);
    for (int i = 0; i < maxIterations; i++) {
        for (int j = 0; j < n; j++) {
            grad[j] = 0;
            for (int k = 0; k < m; k++) {
                h[k] = sigmoid(theta[0] + theta[1] * X[k][1]); // Using sex as predictor
                grad[j] += (h[k] - y[k]) * X[k][j];
            }
            grad[j] /= m;
            theta[j] -= alpha * grad[j];
        }
    }
}

// Prediction function
vector<double> predict(vector<vector<double>> &X, vector<double> &theta) {
    int m = X.size();
    vector<double> y(m);
    for (int i = 0; i < m; i++) {
        y[i] = round(sigmoid(theta[0] + theta[1] * X[i][1])); // Using sex as predictor
    }
    return y;
}

// Accuracy function
double accuracy(vector<double> &yPred, vector<double> &yTrue) {
    int m = yPred.size();
    int count = 0;
    for (int i = 0; i < m; i++) {
        if (yPred[i] == yTrue[i]) {
            count++;
        }
    }
    return (double) count / m;
}

// Sensitivity function
double sensitivity(vector<double> &yPred, vector<double> &yTrue) {
    int m = yPred.size();
    int tp = 0, fn = 0;
    for (int i = 0; i < m; i++) {
        if (yPred[i] == 1 && yTrue[i] == 1) {
            tp++;
        } else if (yPred[i] == 0 && yTrue[i] == 1) {
            fn++;
        }
    }
    return (double) tp / (tp + fn);
}



// Specificity function
double specificity(vector<double> &yPred, vector<double> &yTrue) {
    int m = yPred.size();
    int tn = 0, fp = 0;
    for (int i = 0; i < m; i++) {
        if (yPred[i] == 0 && yTrue[i] == 0) {
            tn++;
        } else if (yPred[i] == 1 && yTrue[i] == 0) {
            fp++;
        }
    }
    return (double) tn / (tn + fp);
}
int main() {
vector<vector<double>> X; // Input data
vector<double> y; // Output data
vector<double> theta(2); // Coefficients
int maxIterations = 1000;
double alpha = 0.01;
ifstream file("titanic_project.csv");
string line;
getline(file, line); // Skip header
while (getline(file, line)) {
vector<double> row;
stringstream ss(line);
string value;
while (getline(ss, value, ',')) {
if (value.size() > 0) {
    try {
        row.push_back(stod(value));
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << ", value: " << value << std::endl;
    }
}

}
X.push_back({1, row[3]}); // Using sex as predictor
y.push_back(row[1]);
}
file.close();
// Split data into training and testing sets
int nTrain = 800;
vector<vector<double>> XTrain(X.begin(), X.begin() + nTrain);
vector<double> yTrain(y.begin(), y.begin() + nTrain);
vector<vector<double>> XTest(X.begin() + nTrain, X.end());
vector<double> yTest(y.begin() + nTrain, y.end());

// Train logistic regression model
vector<double> thetaTrain(2);
logisticRegression(XTrain, yTrain, thetaTrain, maxIterations, alpha);

// Test logistic regression model
vector<double> yPred = predict(XTest, thetaTrain);
double acc = accuracy(yPred, yTest);
double sens = sensitivity(yPred, yTest);
double spec = specificity(yPred, yTest);

// Output metrics
cout << "Coefficients: " << thetaTrain[0] << ", " << thetaTrain[1] << endl;
cout << "Accuracy: " << acc << endl;
cout << "Sensitivity: " << sens << endl;
cout << "Specificity: " << spec << endl;

return 0;
}