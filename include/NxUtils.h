#ifndef _NxUTILS_H_
#define _NxUTILS_H_

#include "NxCore.h"


/// Maximum Number of elements a NxList object can store.
#define NxMAX_LIST_SIZE 512

typedef enum NxNodeType {
	NxNODE_TYPE_DENSE, ///< used for NxDense layer objects.
	NxNODE_TYPE_CONV1D, ///< used for NxConv1D layer objects.
	NxNODE_TYPE_CONV2D, ///< used for NxConv2D layer objects.
	NxNODE_TYPE_MAXPOOL1D, ///< used for NxMaxPool1D layer objects.
	NxNODE_TYPE_MAXPOOL2D, ///< used for NxMaxPool2D layer objects.
}NxNodeType;

/**
 * @brief Represent a Node in the list object.
 */
typedef struct NxListNode {
	NxNodeType itype; ///< the type of the data stored in the node.
	void* data; ///< the data stored in the node as void* to Generic Usage.
	struct NxListNode* next; ///< pointer to the next node in the list default to `NULL`.
}NxListNode;

void NxListNode_create (NxListNode* pnode, void* data, NxNodeType itype);
void NxListNode_free   (NxListNode* pnode);

/**
 * @brief Represent the List Data Strucutre.
 *
 * This strucure is to represent the list data strucuture that will become
 * very handy later in the storage of the layers of the network speciefied
 * by the used. It will make it very easy to retrive, search, or add layers
 * later.
 *
 * This list is quit complex as it may be considered a Generic List. It stores
 * it's data as nodes of object `NxNode` and each Node store it's data 
 * as `void*` and a parameter called `itype` that represent the type of the
 * node each node stored in the list.
 */
typedef struct NxList {
	u64 size;
	NxListNode* head;
}NxList;

void NxList_create       (NxList* plist);
void NxList_append       (NxList* plist, NxListNode* pnode);
void NxList_delete       (NxList* plist, NxListNode* pnode);
void NxList_print        (NxList* plist);
void NxList_free         (NxList* plist);


#endif /* _NxUTILS_H_ */

/****************************************************************************
 * Copyright (C) 2023 by Moaz Mohammed El-Essawey                           *
 *                                                                          *
 * This file is part of Nexum Library.                                      *
 *                                                                          *
 *   Nexum is free software: you can redistribute it and/or modify it       *
 *   under the terms of the GNU Lesser General Public License as published  *
 *   by the Free Software Foundation, either version 3 of the License, or   *
 *   (at your option) any later version.                                    *
 *                                                                          *
 *   Box is distributed in the hope that it will be useful,                 *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          *
 *   GNU Lesser General Public License for more details.                    *
 *                                                                          *
 *   You should have received a copy of the GNU Lesser General Public       *
 *   License along with Box.  If not, see <http://www.gnu.org/licenses/>.   *
 ****************************************************************************/

/**
 * @file NxUtils.h
 * @author Moaz El-Essawey
 * @date 16 July 2023
 */