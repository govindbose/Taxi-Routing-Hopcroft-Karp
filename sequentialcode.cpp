#include <algorithm>
#include <iostream>
#include <bits/stdc++.h>
#include <fstream>
#include <string>
#include <ctime>
using namespace std;

const int MAXN1 = 50000;
const int MAXN2 = 50000;
const int MAXM = 500000;

int n1, n2, edges, last[MAXN1], previous[MAXM], head[MAXM];
int matching[MAXN2], dist[MAXN1], Q[MAXN1];
bool used[MAXN1], vis[MAXN1];

void init(int _n1, int _n2) {
    n1 = _n1;
    n2 = _n2;
    edges = 0;
    fill(last, last + n1, -1);
}

void addEdge(int u, int v) {
    head[edges] = v;
    previous[edges] = last[u];
    last[u] = edges++;
}

void bfs() {
    fill(dist, dist + n1, -1);
    int sizeQ = 0;
    for (int u = 0; u < n1; ++u) {
        if (!used[u]) {
            Q[sizeQ++] = u;
            dist[u] = 0;
        }
    }
    for (int i = 0; i < sizeQ; i++) {
        int u1 = Q[i];
        for (int e = last[u1]; e >= 0; e = previous[e]) {
            int u2 = matching[head[e]];
            if (u2 >= 0 && dist[u2] < 0) {
                dist[u2] = dist[u1] + 1;
                Q[sizeQ++] = u2;
            }
        }
    }
}

bool dfs(int u1) {
    vis[u1] = true;
    for (int e = last[u1]; e >= 0; e = previous[e]) {
        int v = head[e];
        int u2 = matching[v];
        if (u2 < 0 || !vis[u2] && dist[u2] == dist[u1] + 1 && dfs(u2)) {
            matching[v] = u1;
            used[u1] = true;
            return true;
        }
    }
    return false;
}

int maxMatching() {
    fill(used, used + n1, false);
    fill(matching, matching + n2, -1);
    for (int res = 0;;) {
        bfs();
        fill(vis, vis + n1, false);
        int f = 0;
        for (int u = 0; u < n1; ++u)
            if (!used[u] && dfs(u))
                ++f;
        if (!f)
            return res;
        res += f;
    }
}

int main() {
    int nodes=999;
    init(nodes, nodes);
    int readnode[2];
    string line;
    ifstream myfile ("data3.txt");
    if (myfile.is_open())
    {
      while ( getline (myfile,line) )
      {
        std::istringstream is(line);
        int n;
        int i=0;
        while( is >> n ) {
             readnode[i]=n;
             i++;
        }
        //cout<<readnode[0]<<" "<<readnode[1]<<endl;
        addEdge((readnode[0]-1),(readnode[1]-1));
      }
    myfile.close();
    }

	clock_t cpub = clock();
    int maxmat=maxMatching();
	clock_t cpue = clock();
	double cpu_time = 1000* double(cpue - cpub) / CLOCKS_PER_SEC;
	printf("Time :%f ms\n",cpu_time);             	
    cout << "Size of maximum matching is "<<maxmat<<endl; 
    cout << "Minimum size of fleet required is "<<(nodes-maxmat)<<endl; 
    return 0;
}
