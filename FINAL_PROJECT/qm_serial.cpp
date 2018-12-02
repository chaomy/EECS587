#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace std;
int n =16;
bool comp(int n, char* a, char* b){
	for(int i = 0; i<n; i++){
		if(a[i]!=b[i]&&(a[i]!='2'&&b[i]!='2')) return false;
	}
	return true;
}

int imp(int n, char* a, char* b){
	int count = 0;
	int temp;
	for(int i = 0; i<n; i++){
		if(a[i]!=b[i]){
			count ++;
			temp = i;
		}
	}
	if(count == 1) return temp;
	else return -1;
}


int main(){
	int i,j,k,l,count,tmp;
	char *temp,c;
	vector<char*> v[16];
	vector <char*> relative;
	vector <char*> prime;
	vector<char*> result;
	bool flag[2];
//read to v[0]
	for(int i = 0; i<16;i++){
		if(v[i].empty()) break;

		for(int j = 0;j<v[i].size()-1;j++){
			bool* flag = new bool[v[i].size()]; //be able to improve
			flag = 0;
			for(int k = j+1; k<v[i].size();k++){
				int	impt = imp(n,v[i][j],v[i][k]);
				if(impt!=-1){
					flag[j] = 1;
					flag[k] = 1;
					strcpy(temp, v[i][j]);
					temp[impt]='2';
					if(find(v[i+1].begin(),v[i+1].end(),temp)==v[i+1].end())
						v[i+1].push_back(temp);
				}

			}

		}
		for(int j = 0; j<v[i].size(); j++){
			if(!flag[j]){
				prime.push_back(v[i][j]);
			}
		}
	}

	for(int i = 0; i<relative.size(); i++){
		int count = 0;
		int num = 0;
		for(j = 0; j<prime.size(); j++){
			if(comp(n,relative[i],prime[j])){
				count ++;
				num = j;
			}
		}
		if(count == 1){ // essential prime implicant
			result.push_back(prime[num]);

			for(j = 0; j<relative.size();j++){
				if(comp(n, relative[j], prime[num])){
					relative.erase(relative.begin()+j);
					j--;
				}
			}
			prime.erase(prime.begin()+num);
			i--;
		}
	}

	while(!relative.empty()){
		strcpy(temp,prime.back());
		prime.pop_back();
		count = 0;
		for(i = 0; i<relative.size(); i++){
			if(comp(n,relative[i],temp)){
				relative.erase(relative.begin()+i);
				i--;
				count++ ;
			}
		}
		if(count >0){
			result.push_back(temp);
		}
	}

	
	
}
