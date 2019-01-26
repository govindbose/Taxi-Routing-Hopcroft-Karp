#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<ctime>

#define MAX_M 500000
using namespace std;

#define CUDA_CALL(x) do { cudaError_t err=(x); \
	if(err!=cudaSuccess) { \
	printf("Error %s at %s: %d",cudaGetErrorString(err),__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)  

__device__  int getProcessCount(const int nodes){
	/*
	In case the number of threads is less than the 
	total number of nodes in the graph. In this case,
	each thread handles more than one node, and the
	exact number is given by this function
	*/
	int no_threads = gridDim.x*blockDim.x;
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if (tid>=nodes)
		return 0;
	else if (tid< nodes % no_threads)
		return (nodes+no_threads-1)/no_threads;
	else
		return nodes/no_threads;

}


__global__ void fixMatching(int* cmatch, int* rmatch,int *nodes)
{
	/*
	To handle any race conditions that may have arisen.
	We don't explicitly prevent the race conditions, rather
	opting to fix them at each iteration. But each iteration
	guarantees at least one augmenting path hence the number of
	iterations is bounded
	*/
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int process_cnt = getProcessCount(*nodes);
	for(int i=0;i<process_cnt;i++){
		int col_vertex=i*(gridDim.x*blockDim.x)+tid;
		//Race condition
		if(cmatch[rmatch[col_vertex]]!=col_vertex)
			rmatch[col_vertex]=-1;
	}

}

__global__ void initBfsArray(int* bfs_array, int* amatch,int* nodes){
	/*
	Kernel to initialize the BFS array.
	Sets the node to -1 if already matched, 
	and to 0 if the node has not been matched yet
	*/
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int process_cnt = getProcessCount(*nodes);
	for(int i=0;i<process_cnt;i++){
		int col_vertex=i*(gridDim.x*blockDim.x)+tid;
		if(amatch[col_vertex]>-1)
			bfs_array[col_vertex]=-1;
		else if(amatch[col_vertex]==-1)
			bfs_array[col_vertex]=0;
	}
}



__global__ void bfs_edges(int* predecessor, int* bfs_level,int* bfs_array,int* xadj,int* adj,int *nodes, 
		int* rmatch, bool* vertex_inserted, bool* augmenting_path_found, int col_vertex,int start_index)
{
	/*
	Kernel called from the GPU for dynamic parallelism
	Not many changes from the bfs kernel
	*/	   
	int j = threadIdx.x+start_index;
	int neighbour_row=adj[j];
	int col_match=rmatch[neighbour_row];
	if(col_match>-1)
	{
		if(bfs_array[col_match]==-1)
		{
			*vertex_inserted=true;
			bfs_array[col_match]=*bfs_level+1;
			predecessor[neighbour_row]=col_vertex;
		}
	}
	else
	{
		if(col_match==-1)
		{
			rmatch[neighbour_row]=-2;
			predecessor[neighbour_row]=col_vertex;
			*augmenting_path_found=true;
		}
	}
}

__global__ void bfs(int* predecessor, int* bfs_level,int* bfs_array,int* xadj,int* adj,int *nodes, 
		int* rmatch, bool* vertex_inserted, bool* augmenting_path_found)
{
	/*
	Main kernel. Iterates through all the edges
	of a particular node.
	*/
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int process_cnt = getProcessCount(*nodes);
	for(int i=0;i<process_cnt;i++)
	{
		int col_vertex=i*(gridDim.x*blockDim.x)+tid;
		if(bfs_array[col_vertex]==*bfs_level)
		{
			int threads = xadj[col_vertex+1]-xadj[col_vertex];
			//If the node has positive number of neighbours
			if (threads>0)
				bfs_edges<<<1,threads,0>>>( predecessor,bfs_level, bfs_array, xadj, adj,nodes, rmatch,vertex_inserted,augmenting_path_found, col_vertex,xadj[col_vertex]);
		}
	}
}

__global__ void alternate(int* cmatch, int* rmatch, int* nodes,int* predecessor)
{
	/*
	If an augmenting path ending at the vertex 
	has been found, iterate through the predecessor 
	array ie. traverse the augmenting path and alterate 
	the edges to augment it.
	*/
	int tid  = blockIdx.x *blockDim.x+threadIdx.x;
	int process_vent=getProcessCount(*nodes);
	for(int i=0;i<process_vent;i++)
	{
		int row_vertex=i*(gridDim.x*blockDim.x)+tid;
		if(rmatch[row_vertex]==-2)
		{
			while(row_vertex!=-1)
			{
				int matched_col=predecessor[row_vertex];
				int matched_row=cmatch[matched_col];
				if (matched_row!=-1)
					if(predecessor[matched_row]==matched_col)
						break;
				cmatch[matched_col]=row_vertex;
				rmatch[row_vertex]=matched_col;
				row_vertex=matched_row;
			}
		}
	}
}



int main(){

	int *d_predecessor,*d_bfs_level,*d_bfs_array, *d_xadj, *d_adj, *d_amatch,*d_bmatch;

	int nodes= 6511;
    	int edges= 0;
	int readnode[2];
	int n=0;
	int i=0;
	
	int* d_nodes;	
	cudaMalloc(&d_nodes,sizeof(int));
	cudaMemcpy(d_nodes,&nodes,sizeof(int),cudaMemcpyHostToDevice);


	int p=0;
	int index=0;
	int flag1=0;

	// Various timers	
	cudaEvent_t start, stop,kernel_start,kernel_stop;
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_stop);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	float all_time=0 ,kernel_time = 0;
	cudaEventRecord(start,0);	

	//Adjacency list only of first part of bipartite graph	
	int *xadj=(int*)malloc(nodes*sizeof(int));	
	int *adj =(int*)malloc(MAX_M*sizeof(int));
	
	//CPT Time for File IO
	clock_t cpub = clock();
	
	string line;
    	ifstream myfile ("data4.txt");
	/*
	The file should be sorted numerically. use "sort -n"
	1 2
	3 4 
	is valid
	
	3 4
	1 2 
	is not valid
	*/
	
	nodes = nodes+1;
	/*
	FILE IO and conversion into compact adjacency list

	One workaround done here is the increment of node by 1
	for the file io. This is basically adding a phantom node
	and removing it after. The data input file should also have 
	the line 
	<nodes+1> <nodes+1>
	at the end. This was to make the file io and conversion easier.
	The issue if this line doesn't exist can be explained with an
	example. If nodes is 1000 and nodes 999, 998 do not have edges,
	ie they do not turn up in the file, then they will not have
	a legal value set for their xadj and adj.
	*/
	if (myfile.is_open()){
		p=0;
		index=0;
		flag1=0;
		while ( getline (myfile,line) ){
			edges++;
			stringstream S;
		       	S<<line;
			i=0;
			while( S >> n ) {
				readnode[i]=n;
				i++;
			}
			if(!p)
				xadj[p]=0;
			while(p!=readnode[0]-1)
			{
				p++;
				xadj[p]=index;
				if(p>(nodes-1))
				{
					flag1=1;
					break;
				}
			}
			if(flag1)
				break;
			adj[index]=readnode[1]-1;
			index++;

		}
		myfile.close();
	}
	clock_t cpue = clock();
	double cpu_time = 1000* double(cpue - cpub) / CLOCKS_PER_SEC;
	printf("FILE IO Time :%f ms\n",cpu_time);             	
	
	//Removal of the phantom node and edge we added
	nodes = nodes-1;
	edges = edges-1;	

	int *amatch =(int*)malloc(nodes*sizeof(int));	
	int *bmatch =(int*)malloc(nodes*sizeof(int));
	int *bfs_array =(int*)malloc(nodes*sizeof(int));
    
	memset(amatch,-1,sizeof(int)*nodes);
	memset(bmatch,-1,sizeof(int)*nodes);
	
	int bfs_level = 0;
	

	cudaMalloc(&d_amatch,sizeof(int)*nodes);
	cudaMalloc(&d_bmatch,sizeof(int)*nodes);
	cudaMalloc(&d_predecessor,sizeof(int)*nodes);
	cudaMalloc(&d_bfs_level,sizeof(int)*nodes);
	cudaMalloc(&d_bfs_array,sizeof(int)*nodes);
	cudaMalloc(&d_xadj,sizeof(int)*(nodes+1));
	cudaMalloc(&d_adj,sizeof(int)*edges);

	cudaEventRecord(kernel_start,0);	

	//Note the nodes + 1 here. This is so that the final edge doesn't
	//access illegal memory	
	cudaMemcpy(d_xadj, xadj, sizeof(int)*(nodes+1),cudaMemcpyHostToDevice);
	cudaMemcpy(d_adj,adj, sizeof(int)*edges,cudaMemcpyHostToDevice);

	//Number of nodes a thread should handle
	//Default is one
	int nops = 1;
	dim3 threads(4);
	dim3 blocks((nodes+threads.x-1)/(threads.x*nops));

	bool* d_augmenting_path_found;
	bool* d_vertex_inserted;
	bool* augmenting_path_found = (bool*)malloc(sizeof(bool));
	bool* vertex_inserted = (bool*)malloc(sizeof(bool));
	
     	*augmenting_path_found = true;


	cudaMalloc(&d_augmenting_path_found,sizeof(bool));
     	cudaMalloc(&d_vertex_inserted,sizeof(bool));
		
	cudaMemset(d_amatch,-1,sizeof(int)*nodes);
	cudaMemset(d_bmatch,-1,sizeof(int)*nodes);
	
	while (*augmenting_path_found){
	/*
	Main loop. While either an augmenting path was found or
	a new vertex was inserted into the set of vertices for
	consideration.
	
	The program flow mainly involves the three kernels, apart
	from the one used to initialize. Kernel bfs() is the most work
	intensive one. alternate() is only relevant if an augmenting 
	path is found. fixMatching() is a support kernel for fixing
	errors due to race conditions
	*/
		initBfsArray<<<blocks,threads>>> (d_bfs_array,d_amatch,d_nodes);
		
		*vertex_inserted= true;		
		cudaMemset(d_bfs_level,0,sizeof(int));  
		bfs_level = 0;

		while (*vertex_inserted){	
			//Reset flags
			cudaMemset(d_vertex_inserted,false,sizeof(bool));  
			cudaMemset(d_augmenting_path_found,false,sizeof(bool));  
			
			bfs<<<blocks,threads>>> (d_predecessor,d_bfs_level,d_bfs_array,d_xadj,d_adj,d_nodes,d_bmatch,d_vertex_inserted,d_augmenting_path_found);
			
			cudaMemcpy(augmenting_path_found,d_augmenting_path_found,sizeof(bool),cudaMemcpyDeviceToHost);
			cudaMemcpy(vertex_inserted,d_vertex_inserted,sizeof(bool),cudaMemcpyDeviceToHost);

			if (*augmenting_path_found){
				break;
			}
			bfs_level+=1;
			cudaMemcpy(d_bfs_level,&bfs_level,sizeof(int),cudaMemcpyHostToDevice);
		}
		alternate<<<blocks,threads>>> (d_amatch,d_bmatch,d_nodes,d_predecessor);
		fixMatching<<<blocks,threads>>> (d_amatch,d_bmatch,d_nodes);
	}

	cudaMemcpy(amatch,d_amatch, sizeof(int)*nodes,cudaMemcpyDeviceToHost);	
	cudaMemcpy(bmatch,d_bmatch, sizeof(int)*nodes,cudaMemcpyDeviceToHost);	


	cudaEventRecord(kernel_stop,0);
	cudaEventSynchronize(kernel_stop);
	cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);

	//Final check of race conditions
	int maxmat=0;
	for(int i =0;i<nodes;++i){
		if(amatch[i]!=-1)
			if(bmatch[amatch[i]]==i)
				maxmat++;
	}
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&all_time, start, stop);
	
	printf("\nOverall Time : %f ms\nKernel Time :%f ms\n",all_time,kernel_time);
	cout << "Size of maximum matching is "<<maxmat<<endl; 
	cout << "Minimum size of fleet required is "<<(nodes-maxmat)<<endl; 
	free(xadj);
	free(adj);
}

