//========================================================================================================================================================================================================200
//	findRangeK function
//========================================================================================================================================================================================================200

#include <cuda/atomic>

__global__ void 
findRangeK(	long height,

			knode *knodesD,
			long knodes_elem,
			cuda::atomic<long, cuda::thread_scope_device> *currKnodeD,
			//long *currKnodeD,
			cuda::atomic<long, cuda::thread_scope_device> *offsetD,
			//long *offsetD,
			cuda::atomic<long, cuda::thread_scope_device> *lastKnodeD,
			//long *lastKnodeD,
			cuda::atomic<long, cuda::thread_scope_device> *offset_2D,
			//long *offset_2D,
			int *startD,
			int *endD,
			int *RecstartD, 
			int *ReclenD)
{

	cuda::memory_order mem_order = cuda::memory_order_relaxed; 

	// private thread IDs
	int thid = threadIdx.x;
	int bid = blockIdx.x;

	// ???
	int i;
	for(i = 0; i < height; i++){

		if((knodesD[currKnodeD[bid].load(mem_order)].keys[thid] <= startD[bid]) && (knodesD[currKnodeD[bid].load(mem_order)].keys[thid+1] > startD[bid])){
			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[currKnodeD[bid].load(mem_order)].indices[thid] < knodes_elem){
				offsetD[bid].store(knodesD[currKnodeD[bid].load(mem_order)].indices[thid], mem_order);
			}
		}
		if((knodesD[lastKnodeD[bid].load(mem_order)].keys[thid] <= endD[bid]) && (knodesD[lastKnodeD[bid].load(mem_order)].keys[thid+1] > endD[bid])){
			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[lastKnodeD[bid].load(mem_order)].indices[thid] < knodes_elem){
				offset_2D[bid].store(knodesD[lastKnodeD[bid].load(mem_order)].indices[thid], mem_order);
			}
		}
		__syncthreads();

		// set for next tree level
		if(thid==0){
			currKnodeD[bid].store(offsetD[bid].load(mem_order), mem_order);
			lastKnodeD[bid].store(offset_2D[bid].load(mem_order), mem_order);
		}
		__syncthreads();
	}

	// Find the index of the starting record
	if(knodesD[currKnodeD[bid].load(mem_order)].keys[thid] == startD[bid]){
		RecstartD[bid] = knodesD[currKnodeD[bid].load(mem_order)].indices[thid];
	}
	__syncthreads();

	// Find the index of the ending record
	if(knodesD[lastKnodeD[bid]].keys[thid] == endD[bid]){
		ReclenD[bid] = knodesD[lastKnodeD[bid]].indices[thid] - RecstartD[bid]+1;
	}

}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200

// makeAtomic:
// currKnodeD: index dependency (29), written to (49)
// lastKnodeD: index dependency (37), written to (50)
// offsetD: data dependency of currKnodeD (49), written to (34)
// offset_2D: data dependency of lastKnodeD (50), written to (42)
// knode->indices: data dependency of offsetD (34), read only
