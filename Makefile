CC = gcc
CC_FLAGS = -g -Wall
CC_LINKS = -lc -lm

LIB_NAME := libNexum.so
BIN_NAME := Nexum.out
DOCS_CONF := Doxyfile

INC_DIR := include
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
APP_DIR := app
LIB_DIR := lib
TEST_DIR := tests


Nexum_FLAGS = -Iinclude
Nexum_LINKS = -Llib
BLAS_INC = -I/usr/local/include/openblas
BLAS_LINKS = -lopenblas

LIB_TARGET = $(LIB_DIR)/$(LIB_NAME)
BIN_TARGET = $(BIN_DIR)/$(BIN_NAME)

SRCS =                              \
       $(SRC_DIR)/Nexum_Utils.c     \
       $(SRC_DIR)/Nexum_Tensor.c    \
	   $(SRC_DIR)/Nexum_Layers.c    \
	   $(SRC_DIR)/Nexum_Loss.c      \
       $(SRC_DIR)/Nexum_Optimizer.c \
       $(SRC_DIR)/Nexum_Model.c     \
	   
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

TEST_SRCS = $(TEST_DIR)/tests.c

.PHONY: docs
.PHONY: info

all: $(LIB_TARGET)
test: $(LIB_TARGET) $(BIN_TARGET)

run: $(LIB_TARGET) $(BIN_TARGET)
	./$(BIN_TARGET)

$(LIB_TARGET): $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -shared -o $@ $(CC_LINKS) $(BLAS_LINKS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CC_FLAGS) -fPIC -c $< -o $@ $(Nexum_FLAGS) $(BLAS_INC)

$(BIN_TARGET): $(TEST_SRCS)
	$(CC) $(CC_FLAGS) $< -o $@ $(Nexum_FLAGS) $(Nexum_LINKS) $(CC_LINKS) -lNexum 

$(APP_DIR)/%.out: $(APP_DIR)/%.c
	$(CC) $(CC_FLAGS) $< -o $@ $(Nexum_FLAGS) $(Nexum_LINKS) $(CC_LINKS) -lNexum

docs:
	@echo "Generating Docs..."
	@doxygen $(DOCS_CONF)

install: $(LIB_TARGET)
	sudo cp $(LIB_TARGET) /usr/lib/

info:
	@echo "****************************************************************************"
	@echo "*   Nexum is free software: you can redistribute it and/or modify it       *"
	@echo "*   under the terms of the GNU Lesser General Public License as published  *"
	@echo "*   by the Free Software Foundation, either version 3 of the License, or   *"
	@echo "*   (at your option) any later version.                                    *"
	@echo "*                                                                          *"
	@echo "*   Box is distributed in the hope that it will be useful,                 *"
	@echo "*   but WITHOUT ANY WARRANTY; without even the implied warranty of         *"
	@echo "*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          *"
	@echo "*   GNU Lesser General Public License for more details.                    *"
	@echo "*                                                                          *"
	@echo "*   You should have received a copy of the GNU Lesser General Public       *"
	@echo "*   License along with Box.  If not, see <http://www.gnu.org/licenses/>.   *"
	@echo "****************************************************************************"

clean:
	$(RM) $(OBJ_DIR)/*
	$(RM) $(LIB_DIR)/*
	$(RM) $(BIN_DIR)/*
	$(RM) $(APP_DIR)/*.out

