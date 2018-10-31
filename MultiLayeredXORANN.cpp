#include <iostream>
#include <string>
#include <cmath>
#include <ostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <vector>

using namespace std;

/*
* Xhanti Singatha
* SNGXHA002 
* MLLab 7
*/

double sigmoid(double value);
double sumWeights(double input1, double input2, double input3, vector<double> weight);
double primeSigmoid(double net);
vector<string> split_string(string str);

double sigmoid(double value){
       double ans = 1 / ( 1 + exp(-value));
       return ans;
}

double sumWeights(double input1, double input2, double input3, double biasWeight, vector<double> weight){
       double sum = (input1 * weight[0]) + (input2 * weight[1]) + (input3 * weight[2]) + biasWeight;
       return sum;
}

double primeSigmoid(double net){
       return (net * (1 - net));
}

vector<string> split_string(string str){
               int len = str.size();
               vector<string> data;
               
               string str1 = "";
               string item = "";
               
               int counter = 0;
               for (int i = 0; i < len; i++){
                    item = str.at(i);
                    if (item.compare(" ") == 0){
                       data.push_back(str1);
                       str1 = "";
                       counter++;
                    }
                   
                    else if (counter == 3){
                        data.push_back(str.substr(i,len));
                        break;
                    }
                   
                    else {
                        str1 += item;
                    }
                   
               }         
               
               return data;
}
               


int main(){
    
    double lRate = 0.1;
    double bias1 = 0.1;
    double bias2 = 0.1;
    double bias3 = 0.1;
    
    double hiddenBias = 0.1;
    
    fstream myfile;
    myfile.open("training_example.txt");     
    
    // Input layer weights 
    vector<double> inputWeight1 = {1, 1, 1};
    vector<double> inputWeight2 = {1, 1, 1};
    vector<double> inputWeight3 = {1, 1, 1};
    
    // Hidden layer weights
    vector<double> hiddenWeight1 = {1, 1, 1};
    
    string line;
    string x1, x2, x3, y;
    double x_1, x_2, x_3, y_o;
    int a = 0;
    
    for (;;){
         if (myfile.is_open()){
             int counter = 0;
             while (!myfile.eof()){
                    getline(myfile, line);
                    vector<string> copy = split_string(line);
                    
                    x_1 = atof(copy[0].c_str());
                    x_2 = atof(copy[1].c_str());
                    x_3 = atof(copy[2].c_str());
                    y_o = atof(copy[3].c_str());
                    
                    double hiddenNet1 = sumWeights(x_1, x_2, x_3, bias1, inputWeight1);
                    double hiddenNet2 = sumWeights(x_1, x_2, x_3, bias2, inputWeight2);
                    double hiddenNet3 = sumWeights(x_1, x_2, x_3, bias3, inputWeight3);
    
                    double hiddenNode1 = sigmoid(hiddenNet1);
                    double hiddenNode2 = sigmoid(hiddenNet2);
                    double hiddenNode3 = sigmoid(hiddenNet3);
    
                    double primeHidden1 =  primeSigmoid(hiddenNode1);
                    double primeHidden2 =  primeSigmoid(hiddenNode2);
                    double primeHidden3 =  primeSigmoid(hiddenNode3);
    
                    // first forward pass output layer
    
                    double outputNet1 = sumWeights(hiddenNode1, hiddenNode2, hiddenNode3, hiddenBias, hiddenWeight1);
                    double outputNode1 = sigmoid(outputNet1);
                    
                    cout << "Output: " << outputNode1 << " ?= " << y_o <<  "\n";
                    cout << "\n";
    
                    if (outputNode1 == y_o){
                        counter++;
                    }
    
                    else {
                        double primeOutput1 =  primeSigmoid(outputNode1);
                        double outputError1 = (primeOutput1 * (y_o - outputNode1));
       
                        // update output weights 
                        
                        double hiddenError1 = primeHidden1 * (hiddenWeight1[0]*outputError1);
                        double hiddenError2 = primeHidden2 * (hiddenWeight1[1]*outputError1);
                        double hiddenError3 = primeHidden3 * (hiddenWeight1[2]*outputError1);
                        
                        double dweight11 = (lRate * outputError1 * hiddenNode1);
                        double dweight12 = (lRate * outputError1 * hiddenNode2);
                        double dweight13 = (lRate * outputError1 * hiddenNode3);
                        double dbias = (lRate * outputError1 * 1);
       
                        hiddenWeight1[0] = hiddenWeight1[0] + dweight11;
                        hiddenWeight1[1] = hiddenWeight1[1] + dweight12;
                        hiddenWeight1[2] = hiddenWeight1[2] + dweight13;
                        hiddenBias += dbias;
                        
       
                        // update input layer weights

                        double dhweight11 = (lRate * hiddenError1 * x_1);
                        double dhweight12 = (lRate * hiddenError2 * x_1);
                        double dhweight13 = (lRate * hiddenError3 * x_1);
                        double dbias1 = (lRate * hiddenError1 * 1);
       
       
                        double dhweight21 = (lRate * hiddenError1 * x_2);
                        double dhweight22 = (lRate * hiddenError2 * x_2);
                        double dhweight23 = (lRate * hiddenError3 * x_2);
                        double dbias2 = (lRate * hiddenError2 * 1);
                        
                        double dhweight31 = (lRate * hiddenError1 * x_3);
                        double dhweight32 = (lRate * hiddenError2 * x_3);
                        double dhweight33 = (lRate * hiddenError3 * x_3);
                        double dbias3 = (lRate * hiddenError3 * 1);
                        
                        inputWeight1[0] = inputWeight1[0] + dhweight11;
                        inputWeight1[1] = inputWeight1[1] + dhweight12;
                        inputWeight1[2] = inputWeight1[2] + dhweight13;
       
                        inputWeight2[0] = inputWeight2[0] + dhweight21;
                        inputWeight2[1] = inputWeight2[1] + dhweight22;
                        inputWeight2[2] = inputWeight2[2] + dhweight23;
                        
                        inputWeight3[0] = inputWeight3[0] + dhweight31;
                        inputWeight3[1] = inputWeight3[1] + dhweight32;
                        inputWeight3[2] = inputWeight3[2] + dhweight33;
                        
                        bias1 += dbias1;
                        bias2 += dbias2;
                        bias3 += dbias3;
                     }
    
              }
              
              if (counter == 8){
                 break;
                 myfile.close();
              }  
              myfile.clear();
              myfile.seekg(0, ios::beg);
          }
    
          else {
               cout << "Unable to open file";
               break;
          }
          a++;
          cout << a << "\n";
   }
}