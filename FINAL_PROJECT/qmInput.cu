/*
 * @Author: chaomy
 * @Date:   2018-12-01 20:58:53
 * @Last Modified by:   chaomy
 * @Last Modified time: 2018-12-02 00:22:51
 */

#include "qmHome.h"

void QMParallel::readTrueTable(string fname) {
  ifstream s(fname, std::iostream::in);
  string line;

  getline(s, line, ' ');
  getline(s, line);
  in_bit_num = stoi(line); 

  getline(s, line, ' ');
  getline(s, line);  
  out_bit_num = stoi(line); 

  getline(s, line, ' '); 
  getline(s, line); 
  split(line, " ", in_labels); 

  getline(s, line, ' '); 
  getline(s, line); 
  split(line, " ", out_labels); 

  std::copy(in_labels.begin(), in_labels.end(), std::ostream_iterator<string>(std::cout, " "));  
  std::copy(out_labels.begin(), out_labels.end(), std::ostream_iterator<string>(std::cout, " "));

  // read head 
  while (getline(s, line) && (line != ".e")) {  
  	vector<string> buff;  
  	split(line, " ", buff);  
  	input.push_back(buff[0]); 
  	output.push_back(buff[1]);   
  }
  std::copy(input.begin(), input.end(), std::ostream_iterator<string>(std::cout, "\n ")); 
}