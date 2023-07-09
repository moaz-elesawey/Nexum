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
APP_DIR := app


Nexum_FLAGS = -Iinclude
Nexum_LINKS = -Llib
BLAS_INC = -I/usr/local/include/openblas
BLAS_LINKS = -lopenblas

LIB_TARGET = $(LIB_DIR)/$(LIB_NAME)
BIN_TARGET = $(BIN_DIR)/$(BIN_NAME)

SRCS =                              \
       $(SRC_DIR)/Nexum_Tensor.c    \
	   $(SRC_DIR)/Nexum_Dense.c     \
	   $(SRC_DIR)/Nexum_Loss.c      \
       $(SRC_DIR)/Nexum_Optimizer.c \
       $(SRC_DIR)/Nexum_Model.c     \
	   
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

TEST_SRCS = $(TEST_DIR)/tests.c

.PHONY: docs

all: $(LIB_TARGET)
test: $(LIB_TARGET) $(BIN_TARGET)

run: $(LIB_TARGET) $(BIN_TARGET)
	./$(BIN_TARGET)

$(LIB_TARGET): $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -shared -o $@ $(Nexum_LINKS) -lNexum $(CC_LINKS) $(BLAS_LINKS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CC_FLAGS) -fPIC -c $< -o $@ $(Nexum_FLAGS) $(BLAS_INC)

$(BIN_TARGET): $(TEST_SRCS)
	$(CC) $(CC_FLAGS) $< -o $@ $(Nexum_FLAGS) $(CC_LINKS) $(Nexum_LINKS) -lNexum 

$(APP_DIR)/%.out: $(APP_DIR)/%.c
	$(CC) $(CC_FLAGS) $< -o $@ $(Nexum_FLAGS) $(CC_LINKS) $(Nexum_LINKS) -lNexum

docs:
	@echo "Generating Docs..."
	@doxygen $(DOCS_CONF)

install: $(LIB_TARGET)
	sudo cp $(LIB_TARGET) /usr/lib/

clean:
	$(RM) $(OBJ_DIR)/*
	$(RM) $(LIB_DIR)/*
	$(RM) $(BIN_DIR)/*
	$(RM) $(APP_DIR)/*.out

