/*******************************************************************************
* Copyright (c) 2023, STMicroelectronics - All Rights Reserved
*
* This file is part of the VL53L8CX Ultra Lite Driver and is 
* 'BSD 3-clause "New" or "Revised" License' licensed
*
*
* License terms: BSD 3-clause "New" or "Revised" License.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software
* without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*
*******************************************************************************/

#include <fcntl.h> // open()
#include <unistd.h> // close()
#include <time.h> // clock_gettime()

#include <linux/spi/spidev.h>
#include <sys/ioctl.h>

#include "platform.h"
#include "types.h"
#include "vl53l8cx_api.h"

#define VL53L8CX_ERROR_GPIO_SET_FAIL	-1
#define VL53L8CX_COMMS_ERROR		-2
#define VL53L8CX_ERROR_TIME_OUT		-3

#define VL53L8_SPIDEV_CHUNK_SIZE   4096 - 2 // 4096 is the SPIDEV limit, 2 for register index

#define VL53L8CX_SPI_MODE  SPI_MODE_0
#define VL53L8CX_SPI_SPEED_HZ  2000000
#define VL53L8CX_SPI_NB_BITS   8

#define LOG 				printf

static int32_t Linux_SPI_Read_16M(int fd, uint16_t index, uint8_t* read_data, uint32_t read_size, uint32_t speed_hz);
static int32_t Linux_SPI_Write_16M(int fd, uint16_t index, uint8_t* write_data, uint32_t write_size, uint32_t speed_hz);

int32_t vl53l8cx_comms_init(VL53L8CX_Platform * p_platform)
{
	uint8_t spi_mode = VL53L8CX_SPI_MODE;
	uint8_t bits = VL53L8CX_SPI_NB_BITS;
	uint32_t speed = VL53L8CX_SPI_SPEED_HZ;
	char devname[32];
	
	sprintf(devname, "/dev/spidev%d.%d", p_platform->spi_num, p_platform->spi_cs);
	p_platform->fd = open(devname, O_RDWR);
	if (p_platform->fd == -1) {
		LOG("Failed to open %s\n", devname);
		return VL53L8CX_COMMS_ERROR;
	}
	else
		LOG("Opened SPI %s\n", devname);

	if (ioctl(p_platform->fd, SPI_IOC_WR_MODE, &spi_mode) <0) {
		LOG("Could not program clock phase and polarity (WR)\n");
		return VL53L8CX_COMMS_ERROR;
	}

	if (ioctl(p_platform->fd, SPI_IOC_RD_MODE, &spi_mode) <0) {
		LOG("Could not program clock phase and polarity (RD)\n");
		return VL53L8CX_COMMS_ERROR;
	}

	if (ioctl(p_platform->fd, SPI_IOC_WR_BITS_PER_WORD, &bits) <0) {
		LOG("Could not program bits per words (WR)\n");
		return VL53L8CX_COMMS_ERROR;
	}

	if (ioctl(p_platform->fd, SPI_IOC_RD_BITS_PER_WORD, &bits) <0) {
		LOG("Could not program bits per words (RD)\n");
		return VL53L8CX_COMMS_ERROR;
	}

	if (ioctl(p_platform->fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) <0) {
		LOG("Could not program speed (WR)\n");
		return VL53L8CX_COMMS_ERROR;
	}

	if (ioctl(p_platform->fd, SPI_IOC_RD_MAX_SPEED_HZ, &speed) <0) {
		LOG("Could not program speed (RD)\n");
		return VL53L8CX_COMMS_ERROR;
	}

	LOG("Opened ST TOF Dev = %d\n", p_platform->fd);

	return 0;
}

int32_t vl53l8cx_comms_close(VL53L8CX_Platform * p_platform)
{
	close(p_platform->fd);
	return 0;
}

int32_t write_read_multi(
		int fd,
		uint16_t i2c_address,
		uint16_t reg_address,
		uint8_t *pdata,
		uint32_t count,
		int write_not_read)
{
	int32_t status = 0;

	uint32_t data_size = 0;
	uint32_t position = 0;

	if (write_not_read) {
		for (position = 0; position < count; position += VL53L8_SPIDEV_CHUNK_SIZE) {
			if (count > VL53L8_SPIDEV_CHUNK_SIZE) {
				if ((position + VL53L8_SPIDEV_CHUNK_SIZE) > count)
					data_size = count - position;
				else
					data_size = VL53L8_SPIDEV_CHUNK_SIZE;
			} else {
				data_size = count;
			}
			if (Linux_SPI_Write_16M(fd, reg_address + position, pdata + position, data_size, VL53L8CX_SPI_SPEED_HZ) != 0) {
				LOG("vl53l8_write_multi Linux_SPI_Write_16M() failed\n");
				status = VL53L8CX_COMMS_ERROR;
				goto exit;
			}
		}
	}
	else {
		for (position = 0; position < count; position +=
				VL53L8_SPIDEV_CHUNK_SIZE) {
			if (count > VL53L8_SPIDEV_CHUNK_SIZE) {
				if ((position + VL53L8_SPIDEV_CHUNK_SIZE) > count)
					data_size = count - position;
				else
					data_size = VL53L8_SPIDEV_CHUNK_SIZE;
			} else {
				data_size = count;
			}
			if (Linux_SPI_Read_16M(fd, reg_address + position, pdata + position, data_size, VL53L8CX_SPI_SPEED_HZ) != 0) {
				LOG("vl53l8_read_multi: Linux_SPI_Read_16M() failed\n");
				status = VL53L8CX_COMMS_ERROR;
				goto exit;
			}
		}
	}
	
exit:
	return status;
}

int32_t write_multi(
		int fd,
		uint16_t i2c_address,
		uint16_t reg_address,
		uint8_t *pdata,
		uint32_t count)
{
	return(write_read_multi(fd, i2c_address, reg_address, pdata, count, 1));
}

int32_t read_multi(
		int fd,
		uint16_t i2c_address,
		uint16_t reg_address,
		uint8_t *pdata,
		uint32_t count)
{
	return(write_read_multi(fd, i2c_address, reg_address, pdata, count, 0));
}

uint8_t VL53L8CX_RdByte(
		VL53L8CX_Platform * p_platform,
		uint16_t reg_address,
		uint8_t *p_value)
{
	return(read_multi(p_platform->fd, p_platform->address, reg_address, p_value, 1));
}

uint8_t VL53L8CX_WrByte(
		VL53L8CX_Platform * p_platform,
		uint16_t reg_address,
		uint8_t value)
{
	return(write_multi(p_platform->fd, p_platform->address, reg_address, &value, 1));
}

uint8_t VL53L8CX_RdMulti(
		VL53L8CX_Platform * p_platform,
		uint16_t reg_address,
		uint8_t *p_values,
		uint32_t size)
{
	return(read_multi(p_platform->fd, p_platform->address, reg_address, p_values, size));
}

uint8_t VL53L8CX_WrMulti(
		VL53L8CX_Platform * p_platform,
		uint16_t reg_address,
		uint8_t *p_values,
		uint32_t size)
{
	return(write_multi(p_platform->fd, p_platform->address, reg_address, p_values, size));
}

void VL53L8CX_SwapBuffer(
		uint8_t 		*buffer,
		uint16_t 	 	 size)
{
	uint32_t i, tmp;
	
	/* Example of possible implementation using <string.h> */
	for(i = 0; i < size; i = i + 4) 
	{
		tmp = (
		  buffer[i]<<24)
		|(buffer[i+1]<<16)
		|(buffer[i+2]<<8)
		|(buffer[i+3]);
		
		memcpy(&(buffer[i]), &tmp, 4);
	}
}	

uint8_t VL53L8CX_WaitMs(
		VL53L8CX_Platform * p_platform,
		uint32_t time_ms)
{
	usleep(time_ms*1000);
	return 0;
}


/*
 * A write/read SPI transfer with NCS low throughout
 */
static int32_t SPI_Xfer(int device_handle, uint8_t* write_data, uint8_t* read_data, uint32_t len, uint32_t speed_hz)
{
	int32_t error = -1;
	struct spi_ioc_transfer message;

	if(device_handle != 0)
	{
		memset(&message, 0, sizeof(message));

		message.tx_buf = (unsigned long)write_data;
		message.rx_buf = (unsigned long)read_data;
		message.len = len;
		message.delay_usecs = 0;
		message.speed_hz = VL53L8CX_SPI_SPEED_HZ;
		message.bits_per_word = VL53L8CX_SPI_NB_BITS;

		error = ioctl(device_handle, SPI_IOC_MESSAGE(1), &message);

		if (error < 0) {
			LOG("Linux_SPI_Xfer: linux spi error (%d): Write ptr (0x%08x), Read ptr (0x%08x). xfre len (%d), speed_hz(%d)", error, (int)(uintptr_t)write_data, (int)(uintptr_t)read_data, len, message.speed_hz);
			return error;
		} else {
			return 0;
		}
	} else {
		LOG("Linux_SPI_Xfer: attempt to use an invalid handle");
		return -1;
	}
}	


static int32_t Linux_SPI_Read_16M(int fd, uint16_t index, uint8_t* read_data, uint32_t read_size, uint32_t speed_hz)
{
	int32_t status = 0;
	uint8_t* read_buffer = malloc(read_size + 2);
	uint8_t* write_buffer = malloc(read_size + 2);

	/* clear top bit of index - bit 15 is 0 for a read operation on this protocol */
	index = index & 0x7FFF;
	write_buffer[0] = (uint8_t)((index & 0xFF00)>>8);
	write_buffer[1] = (uint8_t)(index & 0xFF);

	status = SPI_Xfer(fd, write_buffer, read_buffer, read_size + 2, speed_hz);

	if (status == 0)
		memcpy(read_data, read_buffer + 2, read_size);

	free(read_buffer);
	free(write_buffer);

	return status;
}

static int32_t Linux_SPI_Write_16M(int fd, uint16_t index, uint8_t* write_data, uint32_t write_size, uint32_t speed_hz)
{
	int32_t status = 0;
	uint8_t* read_buffer = malloc(write_size + 2);
	uint8_t* write_buffer = malloc(write_size + 2);

	write_buffer[0] = (uint8_t)(((index & 0xFF00)>>8 | 0x80));
	write_buffer[1] = (uint8_t)(index & 0xFF);
	memcpy(write_buffer + 2, write_data, write_size);

	status = SPI_Xfer(fd, write_buffer, read_buffer, write_size + 2, speed_hz);

	free(read_buffer);
	free(write_buffer);

	return status;
}

uint8_t VL53L8CX_wait_for_dataready(VL53L8CX_Platform *p_platform)
{
#ifdef STMVL53L8CX_KERNEL
	if (ioctl(p_platform->fd, ST_TOF_IOCTL_WAIT_FOR_INTERRUPT) < 0)
		return 0;
#else
	VL53L8CX_Configuration * p_dev = (VL53L8CX_Configuration *)(p_platform - offsetof(VL53L8CX_Configuration, platform));
	uint8_t isReady = 0;
	do {
		VL53L8CX_WaitMs(p_platform, 5);
		vl53l8cx_check_data_ready(p_dev, &isReady);		
	} while (isReady == 0);
#endif
	return 1;
}