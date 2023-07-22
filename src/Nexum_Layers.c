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
 * @file Nexum_Dense.c
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */


#include "Nexum_Layers.h"

#include <time.h>
#include <math.h>


/**
 * @brief Initialize a new instance of `Nexum_Dense` Structure.
 * 
 * Used to initalize a new object from `Nexum_Dense` Structure and allocates
 * memory for the `weights`, and `bias` terms.
 * 
 * The initialized weights have the shape (in_features, out_features),
 * and the bias term have the shape (out_features, 1). Usually we use 
 * random number for initialization and here I use Normal Random Number for
 * the weights, and the bias is initialized all with zeros.

 * @param L The layer object to be initialized.
 * @param in_features The number of input features.
 * @param out_features The number of output features.
 * @param act The activation function to be used.
 *
 * @return void
 *
 * @see Nexum_Activation, Nexum_Tensor_alloc_randn(), Nexum_Tensor_alloc_zeros().
 */
void Nexum_Dense_alloc(Nexum_Dense* L, u64 in_features, u64 out_features, Nexum_Activation act) {
	L->in_features = in_features;
	L->out_features = out_features;
	L->act = act;
	L->initialized = true;

	Nexum_Tensor_alloc_randn(&(L->weights), out_features, in_features);
	Nexum_Tensor_alloc_zeros(&(L->bias), out_features, 1);
}

void Nexum_Dense_forward(Nexum_Dense* L3, Nexum_Dense* L1, Nexum_Dense* L2) {
	
}

void Nexum_Dense_print(Nexum_Dense* L) {
	Nexum_Tensor_print(&(L->weights));
	Nexum_Tensor_print(&(L->bias));
}

void Nexum_Dense_read(Nexum_Dense* L, str fname) {
	FILE* fptr = fopen(fname, READ_MODE);
	u64 in_features, out_features;
	Nexum_Activation act;
	if(fptr == NULL) {
		return ;
	}
	fscanf(fptr, "%lu %lu %u", &in_features, &out_features, &act);
	Nexum_Dense_alloc(L, in_features, out_features, act);
	for(u64 i=0; i<L->in_features; i++) {
		for(u64 j=0; j<L->out_features; j++) {
			fscanf(fptr, "%lf", &(L->weights.data[L->in_features*i + j]));
		}
	}
	for(u64 j=0; j<L->out_features; j++) {
		fscanf(fptr, "%lf", &(L->bias.data[j]));
	}
	fclose(fptr);
}

void Nexum_Dense_read_binary(Nexum_Dense* L, str fname) {
	FILE* fptr = fopen(fname, READ_BINARY_MODE);
	u64 in_features, out_features;
	Nexum_Activation act;
	if(fptr == NULL) {
		return ;
	}
	fread(&in_features, sizeof (u64), 1, fptr);
	fread(&out_features, sizeof (u64), 1, fptr);
	fread(&act, sizeof (Nexum_Activation), 1, fptr);
	Nexum_Dense_alloc(L, in_features, out_features, act);
	fread(&(L->weights.data[0]), sizeof (f64), in_features*out_features, fptr);
	fread(&(L->bias.data[0]), sizeof (f64), out_features, fptr);
	fclose(fptr);
}

void Nexum_Dense_write(Nexum_Dense* L, str fname) {
	FILE* fptr = fopen(fname, WRITE_MODE);

	if(fptr == NULL) {
		return ;
	}
	fprintf(fptr, "%lu %lu %u\n", L->in_features, L->out_features, L->act);
	for(u64 i=0; i<L->in_features; i++) {
		for(u64 j=0; j<L->out_features; j++) {
			fprintf(fptr, "%.3e ", L->weights.data[L->in_features*i + j]);
		}
		fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n");
	
	for(u64 j=0; j<L->out_features; j++) {
		fprintf(fptr, "%.3e ", L->bias.data[j]);
	}
	fprintf(fptr, "\n");
	fclose(fptr);
}

void Nexum_Dense_write_binary(Nexum_Dense* L, str fname) {
	FILE* fptr = fopen(fname, WRITE_BINARY_MODE);

	if(fptr == NULL) {
		return ;
	}
	fwrite(&(L->in_features), sizeof (u64), 1, fptr);
	fwrite(&(L->out_features), sizeof (u64), 1, fptr);
	fwrite(&(L->act), sizeof (Nexum_Activation), 1, fptr);
	fwrite(&(L->weights.data[0]), sizeof (u64), L->in_features*L->out_features, fptr);
	fwrite(&(L->bias.data[0]), sizeof (u64), L->out_features, fptr);
	fclose(fptr);
}

void Nexum_Dense_free(Nexum_Dense* L) {
	Nexum_Tensor_free(&(L->weights));
	Nexum_Tensor_free(&(L->bias));
	L->initialized = false;
	L->in_features = 0;
	L->out_features = 0;
}
