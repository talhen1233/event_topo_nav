CC := gcc

# Set to TRUE to enable logging functions.
LOG_ENABLE = FALSE

# Set interface mode. Important Notice: SPI platform flavor is only implemented for userspace operations and needs for spi_dev
# to build for SPI type 'make PLTF_DIR=../platform_spi'
PLTF_DIR?=../platform
#PLTF_DIR?=../platform_spi

CORE_INCLUDE_PATHS = -I../uld-driver/inc
PLATFORM_INCLUDE_PATHS = -I$(PLTF_DIR)
EXAMPLES_INCLUDE_PATHS = -I../examples

BASE_CFLAGS = -Wall -Werror -Wno-missing-braces
CFLAGS_RELEASE = -Os -g0
#CFLAGS_RELEASE += -DSTMVL53L8CX_KERNEL

LIB_CORE_SOURCES =\
	$(wildcard ../uld-driver/src/*.c)

LIB_PLATFORM_SOURCES =\
	$(wildcard $(PLTF_DIR)/*.c)

LIB_EXAMPLES_SOURCES =\
	$(wildcard ../examples/*.c)

LIB_SOURCES := $(LIB_CORE_SOURCES) $(LIB_PLATFORM_SOURCES) $(LIB_EXAMPLES_SOURCES)
INCLUDE_PATH = $(CORE_INCLUDE_PATHS) $(PLATFORM_INCLUDE_PATHS) $(EXAMPLES_INCLUDE_PATHS)

CFLAGS = $(BASE_CFLAGS) $(CFLAGS_RELEASE) $(INCLUDE_PATH)

all:
	$(CC) $(CFLAGS) $(LIB_FLAGS) -o menu ./menu.c $(LIB_SOURCES)

ifeq ($(findstring spi,$(PLTF_DIR)),spi)
	$(CC) $(CFLAGS) $(LIB_FLAGS) -o multi ./multi_ranging.c $(LIB_SOURCES)
endif

clean:
	rm -f menu multi
