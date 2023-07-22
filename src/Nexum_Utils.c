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
 * @file Nexum_Utils.c
 * @author Moaz El-Essawey
 * @date 16 July 2023
 */

#include "Nexum_Utils.h"

/**
 * @brief Create and initialize a new list object.
 *
 * This function is used to create an instance of Nexum_List object
 * and store it in plist variable.
 *
 * @param plist pointer to the list to create and initialize.
 */
void Nexum_List_create(Nexum_List* plist) {}

/**
 * @brief Add a new element to the list.
 *
 * Add a new element of to the list from the back.
 *
 * @param plist pointer to the list object.
 * @param pnode pointer to the node object in memory.
 */
void Nexum_List_append(Nexum_List* plist, Nexum_ListNode* pnode) {}

/**
 * @brief Removes an element from the list.
 *
 * This function is quit slow as it uses a linear search algorithm
 * in order to find the element we are looking for.
 *
 * @param plist pointer to the list object.
 * @param pnode pointer to the node object in memory.
 *
 * @todo Re-implement the function to use faster searching algo.
 */
void Nexum_List_delete(Nexum_List* plist, Nexum_ListNode* pnode) {}

/**
 * @brief Free the memory from the list data and nodes.
 *
 * @param plist pointer to the list to free.
 */
void Nexum_List_free(Nexum_List* plist) {}

