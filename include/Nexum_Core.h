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
 * @file Nexum_Core.h
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */

#ifndef _Nexum_CORE_H_
#define _Nexum_CORE_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stddef.h>

/// Whether to show debug messages or not.
#define Nexum_DEBUG
// #undef Nexum_DEBUG

typedef int8_t   i8;    ///< shot hand for `int8_t` type
typedef int16_t  i16;   ///< shot hand for `int16_t` type
typedef int32_t  i32;   ///< shot hand for `int32_t` type
typedef int64_t  i64;   ///< shot hand for `int64_t` type
typedef int8_t   u8;    ///< shot hand for `uint8_t` type
typedef uint16_t u16;   ///< shot hand for `uint16_t` type
typedef uint32_t u32;   ///< shot hand for `uint32_t` type
typedef uint64_t u64;   ///< shot hand for `unt64_t` type
typedef float    f32;   ///< shot hand for `float` type
typedef double   f64;   ///< shot hand for `double` type
typedef char*    str;   ///< shot hand for `char*` type

#define READ_MODE          "r"  ///< reading UTF-8 mode
#define WRITE_MODE         "w"  ///< writing UTF-8 mode
#define READ_BINARY_MODE   "rb" ///< reading binary mode
#define WRITE_BINARY_MODE  "wb" ///< weiting binary mode

#define Nexum_DTYPE f64 ///< The default data type of the tensor object.
#define Nexum_CDEF ///< function definition in Nexum lib.
#define Nexum_ASSERT(expr) assert(expr)  ///< overloading the assert function in c.
#define Nexum_MALLOC(size) ((Nexum_DTYPE*)malloc(size)) ///< overloading the malloc function in c.
#define Nexum_LOOP(i, m) for(i=0; i<m; i++) ///< short hand expr for the ordinary for loop.

/// Raises an error if the func is not implemented yet.
#define Nexum_NOTIMPLEMENTED(...) \
    do { \
        fprintf(stderr, "%s:%d : %s is not implemented yet.\n", \
                __FILE__, __LINE__, __func__); \
        abort(); \
    } while(0)

/// Show error, info messages to the console.
#ifdef Nexum_DEBUG
#define Nexum_MESSAGE(status, msg) \
    do { \
        fprintf(stderr, "[%s] %s in %s:%d : %s \n", \
                status, msg, __FILE__, __LINE__, __func__); \
    } while(0)
#else
#define Nexum_MESSAGE(...)
#endif /* Nexum_DEBUG */

#endif /* _Nexum_CORE_H_ */
