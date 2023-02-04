#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;
double sum(const vector<double> &v) {
    double result = 0;
    for (int i = 0; i < v.size(); i++) {
        result += v[i];
    }
    return result;
}

double mean(const vector<double> &v) {
    return sum(v) / v.size();
}

double median(vector<double> v) {
    sort(v.begin(), v.end());
    if (v.size() % 2 == 0) {
        return (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
    } else {
        return v[v.size() / 2];
    }
}

double range(const vector<double> &v) {
    double minVal = v[0];
    double maxVal = v[0];
    for (int i = 1; i < v.size(); i++) {
        if (v[i] < minVal) {
            minVal = v[i];
        }
        if (v[i] > maxVal) {
            maxVal = v[i];
        }
    }
    return maxVal - minVal;
}

void print_stats(const vector<double> &v) {
    cout << "Sum: " << sum(v) << endl;
    cout << "Mean: " << mean(v) << endl;
    cout << "Median: " << median(v) << endl;
    cout << "Range: " << range(v) << endl;
}

double covar(const vector<double> &v1, const vector<double> &v2) {
    double m1 = mean(v1), m2 = mean(v2);
    double covariance = 0;
    for (int i = 0; i < v1.size(); i++) {
        covariance += (v1[i] - m1) * (v2[i] - m2);
    }
    return covariance / (v1.size() - 1);
}

double var(const vector<double> &v) {
    double m = mean(v);
    double variance = 0;
    for (int i = 0; i < v.size(); i++) {
        variance += (v[i] - m) * (v[i] - m);
    }
    return variance / (v.size() - 1);
}

double cor(const vector<double> &v1, const vector<double> &v2) {
    return covar(v1, v2) / sqrt(var(v1) * var(v2));
}


int main(int argc, char** argv){

ifstream inFS;
string line;
string rm_in, medv_in;
const int MAX_LEN = 1000;
vector<double> rm(MAX_LEN);
vector<double> medv(MAX_LEN);

cout << "Opening file Boston.csv." << endl;

inFS.open("Boston.csv");
if(!inFS.is_open()) {
    cout << "Could not open file Boston.csv" << endl;
    return 1;
}

cout << "Reading line 1" <<endl;
getline(inFS, line);

cout <<"heading : " << line << endl;

int numObservations = 0;
while(inFS.good()) {

    getline(inFS, rm_in, ',');
    getline(inFS, medv_in, '\n');

    rm.at(numObservations) = stof(rm_in);
    medv.at(numObservations) = stof(medv_in);

    numObservations++; 
}

rm.resize(numObservations);
medv.resize(numObservations);
cout << "new length" << rm.size() << endl;

cout << "Closing file Boston.csv" << endl;
inFS.close();

cout << "Number of records: " << numObservations << endl;

cout << "\nStats for rm" << endl;
print_stats(rm);

 cout << "\n Covariance = " << covar(rm, medv) << endl;

 cout << "\n Correlation = " << cor(rm, medv) << endl; 

cout <<"\nProgram terminated." ;

return 0;
}