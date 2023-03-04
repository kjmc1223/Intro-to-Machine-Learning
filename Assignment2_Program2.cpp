#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>



using namespace std;

// Compute mean of a vector
double mean(vector<double>& v) {
    double sum = 0;
    int n = v.size();
    for (int i = 0; i < n; i++) {
        sum += v[i];
    }
    return sum / n;
}

// Compute standard deviation of a vector
double stdev(vector<double>& v) {
    double m = mean(v);
    double sum = 0;
    int n = v.size();
    for (int i = 0; i < n; i++) {
        sum += pow(v[i] - m, 2);
    }
    return sqrt(sum / (n - 1));
}

// Compute Gaussian probability density function
double gaussianPDF(double x, double mean, double stdev) {
    double exponent = -pow(x - mean, 2) / (2 * pow(stdev, 2));
    return exp(exponent) / (sqrt(2 * M_PI) * stdev);
}

// Train Naive Bayes classifier
void train_naive_bayes(vector<vector<double>>& X, vector<double>& y, vector<vector<double>>& class_means, vector<vector<double>>& class_stdevs, vector<double>& class_priors) {
    int m = X.size();
    int n = X[0].size();
    vector<vector<double>> X_survived, X_not_survived;
    for (int i = 0; i < m; i++) {
        if (y[i] == 1) {
            X_survived.push_back(X[i]);
        } else {
            X_not_survived.push_back(X[i]);
        }
    }
    int n_survived = X_survived.size();
    int n_not_survived = X_not_survived.size();
    for (int j = 0; j < n; j++) {
        class_means[0][j] = mean(X_not_survived[j]);
        class_means[1][j] = mean(X_survived[j]);
        class_stdevs[0][j] = stdev(X_not_survived[j]);
        class_stdevs[1][j] = stdev(X_survived[j]);
    }
    class_priors[0] = (double) n_not_survived / m;
    class_priors[1] = (double) n_survived / m;
}

// Predict using Naive Bayes classifier
vector<double> predict_naive_bayes(vector<vector<double>>& X, vector<vector<double>>& class_means, vector<vector<double>>& class_stdevs, vector<double>& class_priors) {
    int m = X.size();
    int n = X[0].size();
    vector<double> y_pred(m);
    for (int i = 0; i < m; i++) {
        double p_not_survived = 1;
        double p_survived = 1;
        for (int j = 0; j < n; j++) {
            double x = X[i][j];
            double mean_not_survived = class_means[0][j];
            double stdev_not_survived = class_stdevs[0][j];
            double mean_survived = class_means[1][j];
            double stdev_survived = class_stdevs[1][j];
            p_not_survived *= gaussianPDF(x, mean_not_survived, stdev_not_survived);
p_survived *= gaussianPDF(x, mean_survived, stdev_survived);
}
p_not_survived *= class_priors[0];
p_survived *= class_priors[1];
y_pred[i] = (p_survived > p_not_survived) ? 1 : 0;
}
return y_pred;
}

int main() {
vector<vector<double>> X; // Input data
vector<double> y; // Output data
vector<vector<double>> class_means(2, vector<double>(3)); // Means for each class and feature
vector<vector<double>> class_stdevs(2, vector<double>(3)); // Standard deviations for each class and feature
vector<double> class_priors(2); // Prior probabilities for each class
ifstream file("titanic_project.csv");
string line;
getline(file, line); // Skip header
while (getline(file, line)) {
stringstream ss(line);
vector<double> row;
string value;
while (getline(ss, value, ',')) {
if (value.size() > 0) {
    try {
        row.push_back(stod(value));
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << '\n';
    }
}

}
X.push_back({row[4], row[1], row[3]}); // Age, Pclass, Sex
y.push_back(row[0]);
}
file.close();
// Split data into training and testing sets
int nTrain = 800;
vector<vector<double>> XTrain(X.begin(), X.begin() + nTrain);
vector<double> yTrain(y.begin(), y.begin() + nTrain);
vector<vector<double>> XTest(X.begin() + nTrain, X.end());
vector<double> yTest(y.begin() + nTrain, y.end());

// Train Naive Bayes classifier
train_naive_bayes(XTrain, yTrain, class_means, class_stdevs, class_priors);

// Predict using Naive Bayes classifier
vector<double> yPred = predict_naive_bayes(XTest, class_means, class_stdevs, class_priors);

// Compute test metrics
int mTest = yTest.size();
int tp = 0, fp = 0, tn = 0, fn = 0;
for (int i = 0; i < mTest; i++) {
    if (yTest[i] == 1 && yPred[i] == 1) {
        tp++;
    } else if (yTest[i] == 0 && yPred[i] == 1) {
        fp++;
    } else if (yTest[i] == 0 && yPred[i] == 0) {
        tn++;
    } else if (yTest[i] == 1 && yPred[i] == 0) {
        fn++;
    }
}
double acc = (double) (tp + tn) / mTest;
double sens = (double) tp / (tp + fn);
double spec = (double) tn / (tn + fp);

// Output metrics
cout << "Accuracy: " << acc << endl;
cout << "Sensitivity: " << sens << endl;
cout << "Specificity: " << spec << endl;

return 0;
}