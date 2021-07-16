#define _USE_MATH_DEFINES
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <tgmath.h>
#include "rapidcsv.h" //using Rapidcsv library for CSV parsing
using namespace std;
using namespace rapidcsv;

/*
CS 4375 Project 1 Naive Bayes
Ramesh Kanakala
*/

double variance(vector<double> v, double mean);
double calc_age_lh(double v, double mean, double var);
vector<double> calc_raw_prob(int pclass, int sex, double age, vector<vector<double>> lh_pcl,
	vector<vector<double>> lh_sex, double survpriori, double deadpriori, vector<double> age_mean, vector<double> age_var); 

int main(int argc, char** argv) {
	//read in csv with rapidcsv lib (header file in project directory)
	Document doc("titanic_project.csv", LabelParams(0, 0));

	vector<double> surv = doc.GetColumn<double>("survived"); //survived vector
	vector<double> pcl = doc.GetColumn<double>("pclass"); //pclass vector
	vector<double> sex = doc.GetColumn<double>("sex"); //sex vector
	vector<double> age = doc.GetColumn<double>("age"); //age vector

	//train vectors
	vector<double> survtr;
	vector<double> pcltr;
	vector<double> sextr;
	vector<double> agetr;
	for (int i = 0; i < 900; i++) {
		survtr.push_back(surv[i]);
		pcltr.push_back(pcl[i]);
		sextr.push_back(sex[i]);
		agetr.push_back(age[i]);
	}
	
	//test vectors
	vector<double> survtst;
	vector<double> pcltst;
	vector<double> sextst;
	vector<double> agetst;
	for (int i = 900; i < 1046; i++) {
		survtst.push_back(surv[i]);
		pcltst.push_back(pcl[i]);
		sextst.push_back(sex[i]);
		agetst.push_back(age[i]);
	}

	//start of algorithm
	auto start = chrono::high_resolution_clock::now(); 

	//calculate apriori probs
	double survsum = 0;
	double deadsum = 0;
	for (int i = 0; i < 900; i++) {
		if (survtr[i] == 1)
			survsum++;
		else
			deadsum++;
	}
	double survpriori = survsum/900; //surv count over total
	double deadpriori = deadsum/900; //dead count over total

	//likelihood for pclass
	vector<vector<double>> lh_pcl{ {0, 0}, {0, 0}, {0, 0} };
	//first counting
	for (int i = 0; i < 900; i++) {
		if (survtr[i] == 0) {//dead
			if (pcltr[i] == 1) {
				lh_pcl[0][0]++;//pcl 1
			}
			else if (pcltr[i] == 2) {//pcl 2
				lh_pcl[1][0]++;
			}
			else if (pcltr[i] == 3) {//pcl 3
				lh_pcl[2][0]++;
			}
		}
		else if (survtr[i] == 1){//surv
			if (pcltr[i] == 1) {//pcl 1
				lh_pcl[0][1]++;
			}
			else if (pcltr[i] == 2) {//pcl 2
				lh_pcl[1][1]++;
			}
			else if (pcltr[i] == 3) {//pcl 3
				lh_pcl[2][1]++;
			}
		}
	}
	//dividing for proportion
	for (vector<vector<int>>::size_type i = 0; i < lh_pcl.size(); i++)
	{
		for (vector<int>::size_type j = 0; j < lh_pcl[i].size(); j++)
		{
			if (j == 0) {//dead
				lh_pcl[i][j] = (lh_pcl[i][j]) / deadsum;
			}
			else {//surv
				lh_pcl[i][j] = (lh_pcl[i][j]) / survsum;
			}
		}
	}

	//likelihood for sex
	vector<vector<double>> lh_sex{ {0, 0}, {0, 0} };
	//first counting
	for (int i = 0; i < 900; i++) {
		if (survtr[i] == 0) {//dead
			if (sextr[i] == 0) {
				lh_sex[0][0]++;//sex 0 female
			}
			else if (sextr[i] == 1) {//sex 1 male
				lh_sex[1][0]++;
			}
		}
		else if (survtr[i] == 1) {//surv
			if (sextr[i] == 0) {//sex 0 female
				lh_sex[0][1]++;
			}
			else if (sextr[i] == 1) {//sex 1 male
				lh_sex[1][1]++;
			}
		}
	}
	//dividing for proportion
	for (vector<vector<int>>::size_type i = 0; i < lh_sex.size(); i++)
	{
		for (vector<int>::size_type j = 0; j < lh_sex[i].size(); j++)
		{
			if (j == 0) {//dead
				lh_sex[i][j] = (lh_sex[i][j]) / deadsum;
			}
			else {//surv
				lh_sex[i][j] = (lh_sex[i][j]) / survsum;
			}
		}
	}

	//likelihood for age
	//first calc mean and var
	vector<double> age_mean{ 0, 0 };
	int deadcnt = 0;
	int survcnt = 0;
	vector<double> age_var{ 0, 0 };
	for (int i = 0; i < 900; i++) {
		if (survtr[i] == 0) {//dead
			age_mean[0] += agetr[i];//sum
			deadcnt++;//count number
		}
		else if (survtr[i] == 1) {//surv
			age_mean[1] += agetr[i];
			survcnt++;
		}
	}
	age_mean[0] = age_mean[0] / deadcnt;//dead age mean
	age_mean[1] = age_mean[1] / survcnt;//surv age mean

	vector<double> deadage;//ages of the dead
	vector<double> survage;//ages of the surv
	for (int i = 0; i < 900; i++) {
		if (surv[i] == 0) {//dead
			deadage.push_back(agetr[i]);
		}
		else if (surv[i] == 1) {//surv
			survage.push_back(agetr[i]);
		}
	}
	age_var[0] = variance(deadage, age_mean[0]);//using variance function to calc
	age_var[1] = variance(survage, age_mean[1]);

	//end of algorithm
	auto stop = chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_sec = stop - start;
	cout << endl << "Time: " << elapsed_sec.count() << endl << endl;

	//cout << "Prob. Surv.    Prob. Perish." << endl;
	double probsurv, probdead; //to hold probs
	vector<double> pred; //vector for pred vals
	for (int i = 0; i < 146; i++) {
		//calc probs
		probsurv = calc_raw_prob(int(pcltst[i] - 1), int(sextst[i]), agetst[i], lh_pcl, lh_sex, survpriori, deadpriori, age_mean, age_var)[0];
		probdead = calc_raw_prob(int(pcltst[i] - 1), int(sextst[i]), agetst[i], lh_pcl, lh_sex, survpriori, deadpriori, age_mean, age_var)[1];
		//cout << i+1 << ": " << probsurv << "       " << probdead << endl; //print probs for each instance
		//add to pred 
		if (probsurv > .5)
			pred.push_back(1);
		else
			pred.push_back(0);
	}

	//print lh_pcl for verification
	cout << "lh_pcl" << endl;
	for (vector<vector<int>>::size_type i = 0; i < lh_pcl.size(); i++)
	{
		for (vector<int>::size_type j = 0; j < lh_pcl[i].size(); j++)
		{
			cout << i << j << ": " << lh_pcl[i][j] << endl;
		}
	}
	cout << endl;
	//print lh_sex for verification
	cout << "lh_sex" << endl;
	for (vector<vector<int>>::size_type i = 0; i < lh_pcl.size(); i++)
	{
		for (vector<int>::size_type j = 0; j < lh_pcl[i].size(); j++)
		{
			cout << i << j << ": " << lh_pcl[i][j] << endl;
		}
	}
	cout << endl;

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

//variance function
double variance(vector<double> v, double mean)
{
	double sum = 0.0;
	double temp = 0.0;
	double var = 0.0;
	for (int j = 0; j <= int(v.size()-1); j++)
	{ 
		temp = pow((v[j] - mean), 2);
		sum += temp;
	}
	return var = sum / (v.size() - 1);
}

//age likelihood function
double calc_age_lh(double v, double mean, double var) {
	double lh =  1 / sqrt(2.0 * M_PI * var) * exp(-(pow((v - mean), 2)) / (2.0 * var));
	return lh;
}

//function to calculate Bayes' theorem
vector<double> calc_raw_prob(int pclass, int sex, double age, vector<vector<double>> lh_pcl, 
	vector<vector<double>> lh_sex, double survpriori, double deadpriori, vector<double> age_mean, vector<double> age_var) {
	//numerator surv
	double num_s = lh_pcl[pclass][1] * lh_sex[sex][1] * survpriori * calc_age_lh(age, age_mean[1], age_var[1]);
	//numerator dead
	double num_p = lh_pcl[pclass][0] * lh_sex[sex][0] * deadpriori * calc_age_lh(age, age_mean[0], age_var[0]);
	//denominator
	double denom = num_s + num_p;
	vector<double> rawprob{num_s / denom, num_p / denom };//{surv raw prob, dead raw prob}
	return rawprob;
}