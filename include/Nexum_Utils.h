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
 * @file Nexum_Utils.h
 * @author Moaz El-Essawey
 * @date 16 July 2023
 */


#ifndef _Nexum_UTILS_H_
#define _Nexum_UTILS_H_

#include "Nexum_Core.h"


/// Maximum Number of elements a Nexum_List object can store.
#define Nexum_MAX_LIST_SIZE 512

typedef enum Nexum_NodeType {
	Nexum_NODE_TYPE_DENSE, ///< used for Nexum_Dense layer objects.
	Nexum_NODE_TYPE_CONV1D, ///< used for Nexum_Conv1D layer objects.
	Nexum_NODE_TYPE_CONV2D, ///< used for Nexum_Conv2D layer objects.
	Nexum_NODE_TYPE_MAXPOOL1D, ///< used for Nexum_MaxPool1D layer objects.
	Nexum_NODE_TYPE_MAXPOOL2D, ///< used for Nexum_MaxPool2D layer objects.
}Nexum_NodeType;

/**
 * @brief Represent a Node in the list object.
 */
typedef struct Nexum_ListNode {
	Nexum_NodeType itype; ///< the type of the data stored in the node.
	void* data; ///< the data stored in the node as void* to Generic Usage.
	struct Nexum_ListNode* next; ///< pointer to the next node in the list default to `NULL`.
}Nexum_ListNode;

void Nexum_ListNode_create (Nexum_ListNode* pnode, void* data, Nexum_NodeType itype);
void Nexum_ListNode_free   (Nexum_ListNode* pnode);

/**
 * @brief Represent the List Data Strucutre.
 *
 * This strucure is to represent the list data strucuture that will become
 * very handy later in the storage of the layers of the network speciefied
 * by the used. It will make it very easy to retrive, search, or add layers
 * later.
 *
 * This list is quit complex as it may be considered a Generic List. It stores
 * it's data as nodes of object `Nexum_Node` and each Node store it's data 
 * as `void*` and a parameter called `itype` that represent the type of the
 * node each node stored in the list.
 */
typedef struct Nexum_List {
	u64 size;
	Nexum_ListNode* head;
}Nexum_List;

void Nexum_List_create       (Nexum_List* plist);
void Nexum_List_append       (Nexum_List* plist, Nexum_ListNode* pnode);
void Nexum_List_delete       (Nexum_List* plist, Nexum_ListNode* pnode);
void Nexum_List_print        (Nexum_List* plist);
void Nexum_List_free         (Nexum_List* plist);


#endif /* _Nexum_UTILS_H_ */

