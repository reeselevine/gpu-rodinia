//========================================================================================================================================================================================================200
//	findK function
//========================================================================================================================================================================================================200

#include <cuda/atomic>

__global__ void 
findK(	long height,
		knode *knodesD, 
		long knodes_elem,
		record *recordsD, 
		cuda::atomic<long, cuda::thread_scope_device> *currKnodeD,
		//long *currKnodeD, 
		cuda::atomic<long, cuda::thread_scope_device> *offsetD,
		//long *offsetD, 
		int *keysD,  
		record *ansD) 
{
	cuda::memory_order mem_order = cuda::memory_order_relaxed; 

	// private thread IDs
	int thid = threadIdx.x; 
	int bid = blockIdx.x; 

	// processtree levels
	int i; 
	for(i = 0; i < height; i++){

		// if value is between the two keys
		if((knodesD[currKnodeD[bid].load(mem_order)].keys[thid]) <= keysD[bid] && (knodesD[currKnodeD[bid].load(mem_order)].keys[thid+1] > keysD[bid])){ 
			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset[bid]" calculated below that addresses knodes[] in the next iteration goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[offsetD[bid].load(mem_order)].indices[thid] < knodes_elem){ 
				offsetD[bid].store(knodesD[offsetD[bid].load(mem_order)].indices[thid], mem_order);
			}
		}
		__syncthreads();

		// set for next tree level
		if(thid==0){
			currKnodeD[bid].store(offsetD[bid].load(mem_order), mem_order);
		}
		__syncthreads();

	}

	//At this point, we have a candidate leaf node which may contain
	//the target record.  Check each key to hopefully find the record
	if(knodesD[currKnodeD[bid].load(mem_order)].keys[thid] == keysD[bid]){
		ansD[bid].value = recordsD[knodesD[currKnodeD[bid].load(mem_order)].indices[thid]].value;
	}

}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200

// makeAtomic:
// currKnodeD: index dependency (26), written to (39)
// offsetD: index dependency (31), written to (32)
// knode->indices: data dependency of offsetD (32), read only
