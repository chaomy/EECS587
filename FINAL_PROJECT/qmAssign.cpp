/*
 * @Author: chaomy
 * @Date:   2018-12-09 22:47:41
 * @Last Modified by:   chaomy
 * @Last Modified time: 2018-12-10 01:25:55
 */

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

int calculateAssignMent(int bit_num, int worker_num) {
  int bucket_size = bit_num + 1;
  vector<uint64_t> slots(bucket_size);

  slots[0] = 1;
  for (int i = 1; i < bucket_size; ++i) {
    slots[i] = slots[i - 1] * (bit_num + 1 - i) / i;
  }

  uint64_t total{0};
  for (int i = 0; i < bucket_size - 1; ++i) {
    slots[i] *= slots[i + 1];
    cout << slots[i] << endl;
    total += slots[i];
  }
  total += slots.back();

  double ave = double(total) / worker_num;

  vector<int> res;
  double sum{0};
  for (int i = 0; i < bucket_size - 1; ++i) {
    if (sum + slots[i + 1] >= ave) {
      sum = 0;
      res.push_back(i);
    }
    sum += slots[i];
  }
  copy(res.begin(), res.end(), std::ostream_iterator<int>(cout, " "));
  cout << "ave is " << ave << endl;
}

int main() {
  int bit_num = 28;
  int worker_num = 7;
  calculateAssignMent(bit_num, worker_num);
}