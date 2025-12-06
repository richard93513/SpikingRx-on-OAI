CC = gcc
CFLAGS = -O2 -Wall -g
INCLUDES = -I./ldpc -I./ldpc/common -I./ldpc/openair1/PHY

# 找出所有 ldpc 目錄底下的 .c 檔案
LDPC_SRCS = $(wildcard ldpc/*.c) \
            $(wildcard ldpc/nrLDPC_decoder/*.c) \
            $(wildcard ldpc/nrLDPC_encoder/*.c) \
            $(wildcard ldpc/nrLDPC_coding/*/*.c)

all: spx_ldpc_test

spx_ldpc_test: spx_ldpc_test.c $(LDPC_SRCS)
	$(CC) -o spx_ldpc_test spx_ldpc_test.c $(LDPC_SRCS) $(CFLAGS) $(INCLUDES) -lm

clean:
	rm -f spx_ldpc_test

