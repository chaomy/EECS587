/*
 * @Author: chaomy
 * @Date:   2018-12-09 22:47:41
 * @Last Modified by:   chaomy
 * @Last Modified time: 2018-12-10 14:26:39
 */

#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

/*
  divide nums into K groups such that, the maximum sum of all group is minimized
*/

uint64_t findMinMaxCutting(vector<uint64_t>& nums, int K) {
  int N{(int)nums.size()};

  // dp[i][j]  => prefix [0, i] is cut by j times
  vector<vector<uint64_t>> dp(
      N, vector<uint64_t>(K, std::numeric_limits<uint64_t>::max()));

  // cut 0 times, (no cut)
  uint64_t total = 0;
  for (int i = 0; i < N; ++i) dp[i][0] = (total += nums[i]);

  for (int i = 1; i < N; ++i) {
    int min_cut = std::min(i, K - 1);
    for (int j = 1; j <= min_cut; ++j) {
      uint64_t sum = 0;
      for (int k = i; k > j - 1; --k) {
        if ((sum += nums[k]) > dp[i][j]) break;
        dp[i][j] = std::min(dp[i][j], std::max(dp[k - 1][j - 1], sum));
      }
    }
  }
  return dp[N - 1][K - 1];
}

int calculateAssignMent(int bit_num, int worker_num) {
  int bucket_size = bit_num + 1;
  vector<uint64_t> slots(bucket_size);

  slots[0] = 1;
  for (int i = 1; i < bucket_size; ++i)
    slots[i] = slots[i - 1] * (bit_num + 1 - i) / i;

  for (int i = 0; i < bucket_size - 1; ++i) slots[i] *= slots[i + 1];

  vector<int> res({0});

  uint64_t max_load = findMinMaxCutting(slots, worker_num);
  uint64_t sum{0};

  for (int i = 1; i < bucket_size; sum += slots[i], ++i) {
    if (sum + slots[i] > max_load) {
      sum = 0;
      res.push_back(i);
    }
  }

  cout << " bucket size = " << bucket_size << "max load  " << max_load << endl;
  copy(res.begin(), res.end(), std::ostream_iterator<int>(cout, " "));
}

int main(int argc, char** argv) {
  int bit_num = atoi(argv[1]);
  int worker_num = atoi(argv[2]);
  calculateAssignMent(bit_num, worker_num);
}