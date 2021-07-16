#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <tgmath.h>
#include "rapidcsv.h" //using Rapidcsv library for CSV parsing
using namespace std;
using namespace rapidcsv;

/*
CS 4375 Project 1 Logistic Regression
Ramesh Kanakala
*/

void sigmoid(double pcldata[], double weights[], double sigvec[]);

int main(int argc, char** argv) {
	//read in csv with rapidcsv lib (header file in project directory)
	Document doc("titanic_project.csv", LabelParams(0, 0));

	//tried to work with vectors and eigen matrix multiplications the first time
	//but ran into long run times and many errors so restarted with arrays and 
	//more simple calculations; copied my original vectors into arrays below
	vector<double> surv = doc.GetColumn<double>("survived"); //survived vector
	vector<double> pcl = doc.GetColumn<double>("pclass"); //pclass vector

	//train arrays
	double survtr[900]; //survived train
	double pcltr[900]; //pclass train
	for (int i = 0; i < 900; i++) {
		survtr[i] = surv.at(i);
		pcltr[i] = pcl.at(i);
	}

	//test arrays
	double survtst[146]; //survived test
	double pcltst[146]; //pclass train
	for (int i = 900; i < 1046; i++) {
		survtst[i-900] = surv.at(i);
		pcltst[i-900] = pcl.at(i);
	}

	double weights[2] = { 1, 1 }; //start weights
	double sigvec[900]; //sigmoid vector of probs
	double errors[900]; //errors
	double lrnrt = .001; //learning rate

	//start of algorithm
	auto start = chrono::high_resolution_clock::now();
	
	for (int i = 0; i < 5000; i++) {
		//1. multiply data by weights to get log lh and run thru sigmoid for probs
		sigmoid(pcltr, weights, sigvec);
		//2. compute error, the true values, survtr, minus the probs, sigvec
		for (int j = 0; j < 900; j++) {  
			errors[j] = survtr[j] - sigvec[j]; 
		}
		//3. update weights
		double grad[2] = { 0, 0 };
		//gradient is the X values times the errors
		for (int i = 0; i < 900; i++) {
			grad[0] = grad[0] + errors[i]; //just 1 * error
			grad[1] = grad[1] + (pcltr[i] * errors[i]); //pclass * error
		}
		//updates by adding lrnrt * gradient;
		for (int i = 0; i < 2; i++) {
			weights[i] = weights[i] + (lrnrt * grad[i]);
		}
	}

	//end of algorithm
	auto stop = chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_sec = stop - start;
	cout << endl << "Time: " << elapsed_sec.count() << endl << endl;
	cout << "w0" << ": " << weights[0] << "  w1: " << weights[1] << endl << endl;

	//predict with weights
	double pred[900];
	double prob = 0, survlodds = 0;
	for (int i = 0; i < 146; i++) {
		survlodds = weights[0] + (weights[1] * pcltst[i]); //log oods = linear comb of w0 + w1x
		prob = (1 / (1 + exp(-1 * survlodds))); //prob
		if (prob < 0.5)
			pred[i] = 0;
		else
			pred[i] = 1;
	}

	//calc components of confusion matrix
	double TP = 0, FN = 0, FP = 0, TN = 0;
	for (int i = 0; i < 146; i++) {
		if (survtst[i] == 0 && pred[i] == 0)
			TP++;
		else if (survtst[i] == 1 && pred[i] == 0)
			FP++;
		else if (survtst[i] == 1 && pred[i] == 1)
			TN++;
		else if (survtst[i] == 0 && pred[i] == 1)
			FN++;
	}

	cout << "TP: " << TP << endl;
	cout << "FN: " << FN << endl;
	cout << "FP: " << FP << endl;
	cout << "TN: " << TN << endl << endl;

	//accuracy, sensitivity, specificity
	double accuracy = (TP + TN) / (TP + TN + FP + FN);
	double sensitivity = (TP) / (TP + FN);
	double specificity = (TN) / (TN + FP);

	cout << "Accuracy: " << accuracy << endl;
	cout << "Sensitivity: " << sensitivity << endl;
	cout << "Specificity: " << specificity << endl;
}

//sigmoid/logistic function that will take an input matrix 
//and return a vector of sigmoid values for each observation
void sigmoid(double pcldata[], double weights[], double sigvec[]) {
	double w0inp = 0, w1inp = 0, z = 0, res = 0;
	for (int i = 0; i < 900; i++) {
		w0inp =  1.0 * weights[0]; //intercept mult by 1
		w1inp = pcldata[i] * weights[1]; //pclass mult by w1
		z = w0inp + w1inp; //log lh inp
		res = 1.0 / (1.0 + exp(z * -1.0)); //sigmoid
		sigvec[i] = res; //add to prob vec
	}
}