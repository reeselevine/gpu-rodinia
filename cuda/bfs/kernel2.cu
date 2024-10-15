/*********************************************************************************
Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
All rights reserved.
  
Permission to use, copy, modify and distribute this software and its documentation for 
educational purpose is hereby granted without fee, provided that the above copyright 
notice and this permission notice appear in all copies of this software and that you do 
not sell the software.
  
THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
OTHERWISE.

The CUDA Kernel for Applying BFS on a loaded Graph. Created By Pawan Harish
**********************************************************************************/
#ifndef _KERNEL2_H_
#define _KERNEL2_H_

#include <cuda/atomic>

__global__ void
Kernel2( cuda::atomic<bool, cuda::thread_scope_device>* g_graph_mask, cuda::atomic<bool, cuda::thread_scope_device> *g_updating_graph_mask, bool* g_graph_visited, bool *g_over, int no_of_nodes)
{
	cuda::memory_order mem_order = cuda::memory_order_relaxed; 
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if( tid<no_of_nodes && g_updating_graph_mask[tid].load(mem_order))
	{

		g_graph_mask[tid].store(true, mem_order);
		g_graph_visited[tid]=true;
		*g_over=true;
		g_updating_graph_mask[tid].store(false, mem_order);
	}
}

#endif

// makeAtomic:
// g_updating_graph_mask: control dependency (25), written to (31)

