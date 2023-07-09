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

#define READ_MODE          "r"
#define WRITE_MODE         "w"
#define READ_BINARY_MODE   "rb"
#define WRITE_BINARY_MODE  "wb"

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

#endif /* _Nexum_CORE_H_ */
