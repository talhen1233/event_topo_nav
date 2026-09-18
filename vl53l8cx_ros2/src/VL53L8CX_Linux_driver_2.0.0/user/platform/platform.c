/**
  * platform.c
  * Modified to allow changing I2C address and I2C device path via environment variables.
  *
  * Environment variables:
  *   VL53L8CX_I2C_DEV      => e.g. "/dev/i2c-1"
  *   VL53L8CX_I2C_ADDRESS  => e.g. "0x54"   (for a sensor at 7-bit addr 0x2A)
  *
  * If not set, defaults to "/dev/i2c-4" and address "0x52" (7-bit 0x29).
  */

  #include <fcntl.h>     // open()
  #include <unistd.h>    // close()
  #include <time.h>      // clock_gettime()
  #include <string.h>    // memcpy
  #include <stdlib.h>    // getenv(), strtol()
  #include <linux/i2c.h>
  #include <linux/i2c-dev.h>
  #include <sys/ioctl.h>
  
  #include "platform.h"
  #include "types.h"
  #include "vl53l8cx_api.h"
  
  // Some error codes
  #define VL53L8CX_ERROR_GPIO_SET_FAIL    -1
  #define VL53L8CX_COMMS_ERROR            -2
  #define VL53L8CX_ERROR_TIME_OUT         -3
  
  #define VL53L8CX_COMMS_CHUNK_SIZE       1024
  #define LOG                              printf
  
  #ifndef STMVL53L8CX_KERNEL
  static uint8_t i2c_buffer[VL53L8CX_COMMS_CHUNK_SIZE];
  #else
  struct comms_struct {
	  uint16_t   len;
	  uint16_t   reg_address;
	  uint8_t    write_not_read;
	  uint8_t    padding[3]; /* 64bits alignment */
	  uint64_t   bufptr;
  };
  #endif
  
  #define ST_TOF_IOCTL_TRANSFER           _IOWR('a',0x1, struct comms_struct)
  #define ST_TOF_IOCTL_WAIT_FOR_INTERRUPT _IO('a',0x2)
  
  /**
   * Initialize I2C communication channel.
   */
  int32_t vl53l8cx_comms_init(VL53L8CX_Platform * p_platform)
  {
	  const char *env_i2c_dev = getenv("VL53L8CX_I2C_DEV");
	  const char *dev_path = "/dev/i2c-0";
	  printf("PLATFORM VERSION: HELLO FROM platform.c\n");
	  if (env_i2c_dev != NULL) {
		  dev_path = env_i2c_dev;
	  }
	  printf("[DEBUG] platform.c opening I2C dev path: %s\n", dev_path);
  
	  const char *env_i2c_addr = getenv("VL53L8CX_I2C_ADDRESS");
	  uint16_t custom_addr = 0x52;
	  if (env_i2c_addr != NULL) {
		  custom_addr = (uint16_t)strtol(env_i2c_addr, NULL, 0);
	  }
	  p_platform->address = custom_addr;
  
	  printf("[DEBUG] platform.c using I2C address: 0x%X\n", custom_addr);
  
	  p_platform->fd = open(dev_path, O_RDWR);
	  if (p_platform->fd == -1) {
		  perror("[ERROR] open() failed");
		  return VL53L8CX_COMMS_ERROR;
	  }
  
	  if (ioctl(p_platform->fd, I2C_SLAVE, p_platform->address) < 0) {
		  perror("[ERROR] ioctl(I2C_SLAVE) failed");
		  close(p_platform->fd);
		  return VL53L8CX_COMMS_ERROR;
	  }
  
	  printf("[DEBUG] platform.c opened I2C successfully.\n");
	  return 0;
  }
  
  
  /**
   * Close the I2C communication channel.
   */
  int32_t vl53l8cx_comms_close(VL53L8CX_Platform * p_platform)
  {
	  close(p_platform->fd);
	  return 0;
  }
  
  /**
   * Helper for write_multi and read_multi. 
   */
  int32_t write_read_multi(
	  int fd,
	  uint16_t i2c_address,
	  uint16_t reg_address,
	  uint8_t *pdata,
	  uint32_t count,
	  int write_not_read)
  {
  #ifdef STMVL53L8CX_KERNEL
	  struct comms_struct cs;
  
	  cs.len = count;
	  cs.reg_address = reg_address;
	  cs.bufptr = (uint64_t)(uintptr_t)pdata;
	  cs.write_not_read = (uint8_t)write_not_read;
  
	  if (ioctl(fd, ST_TOF_IOCTL_TRANSFER, &cs) < 0)
		  return VL53L8CX_COMMS_ERROR;
  
  #else
	  struct i2c_rdwr_ioctl_data packets;
	  struct i2c_msg messages[2];
  
	  uint32_t data_size = 0;
	  uint32_t position = 0;
  
	  if (write_not_read) {
		  // Writing
		  do {
			  data_size = ((count - position) > (VL53L8CX_COMMS_CHUNK_SIZE - 2))
						  ? (VL53L8CX_COMMS_CHUNK_SIZE - 2)
						  : (count - position);
  
			  memcpy(&i2c_buffer[2], &pdata[position], data_size);
  
			  i2c_buffer[0] = (uint8_t)((reg_address + position) >> 8);
			  i2c_buffer[1] = (uint8_t)((reg_address + position) & 0xFF);
  
			  messages[0].addr  = (i2c_address >> 1);
			  messages[0].flags = 0; // I2C_M_WR;
			  messages[0].len   = data_size + 2;
			  messages[0].buf   = i2c_buffer;
  
			  packets.msgs  = messages;
			  packets.nmsgs = 1;
  
			  if (ioctl(fd, I2C_RDWR, &packets) < 0)
				  return VL53L8CX_COMMS_ERROR;
  
			  position += data_size;
		  } while (position < count);
  
	  } else {
		  // Reading
		  do {
			  data_size = ((count - position) > VL53L8CX_COMMS_CHUNK_SIZE)
						  ? VL53L8CX_COMMS_CHUNK_SIZE
						  : (count - position);
  
			  i2c_buffer[0] = (uint8_t)((reg_address + position) >> 8);
			  i2c_buffer[1] = (uint8_t)((reg_address + position) & 0xFF);
  
			  messages[0].addr  = (i2c_address >> 1);
			  messages[0].flags = 0; // I2C_M_WR;
			  messages[0].len   = 2;
			  messages[0].buf   = i2c_buffer;
  
			  messages[1].addr  = (i2c_address >> 1);
			  messages[1].flags = I2C_M_RD;
			  messages[1].len   = data_size;
			  messages[1].buf   = &pdata[position];
  
			  packets.msgs  = messages;
			  packets.nmsgs = 2;
  
			  if (ioctl(fd, I2C_RDWR, &packets) < 0)
				  return VL53L8CX_COMMS_ERROR;
  
			  position += data_size;
		  } while (position < count);
	  }
  #endif
	  return 0;
  }
  
  /**
   * Write multiple bytes to the sensor.
   */
  int32_t write_multi(
	  int fd,
	  uint16_t i2c_address,
	  uint16_t reg_address,
	  uint8_t *pdata,
	  uint32_t count)
  {
	  return write_read_multi(fd, i2c_address, reg_address, pdata, count, 1);
  }
  
  /**
   * Read multiple bytes from the sensor.
   */
  int32_t read_multi(
	  int fd,
	  uint16_t i2c_address,
	  uint16_t reg_address,
	  uint8_t *pdata,
	  uint32_t count)
  {
	  return write_read_multi(fd, i2c_address, reg_address, pdata, count, 0);
  }
  
  /**
   * API function: read single byte
   */
  uint8_t VL53L8CX_RdByte(
	  VL53L8CX_Platform * p_platform,
	  uint16_t reg_address,
	  uint8_t *p_value)
  {
	  return (uint8_t)read_multi(p_platform->fd, p_platform->address, reg_address, p_value, 1);
  }
  
  /**
   * API function: write single byte
   */
  uint8_t VL53L8CX_WrByte(
	  VL53L8CX_Platform * p_platform,
	  uint16_t reg_address,
	  uint8_t value)
  {
	  return (uint8_t)write_multi(p_platform->fd, p_platform->address, reg_address, &value, 1);
  }
  
  /**
   * API function: read multiple bytes
   */
  uint8_t VL53L8CX_RdMulti(
	  VL53L8CX_Platform * p_platform,
	  uint16_t reg_address,
	  uint8_t *p_values,
	  uint32_t size)
  {
	  return (uint8_t)read_multi(p_platform->fd, p_platform->address, reg_address, p_values, size);
  }
  
  /**
   * API function: write multiple bytes
   */
  uint8_t VL53L8CX_WrMulti(
	  VL53L8CX_Platform * p_platform,
	  uint16_t reg_address,
	  uint8_t *p_values,
	  uint32_t size)
  {
	  return (uint8_t)write_multi(p_platform->fd, p_platform->address, reg_address, p_values, size);
  }
  
  /**
   * Swap buffer utility
   */
  void VL53L8CX_SwapBuffer(uint8_t *buffer, uint16_t size)
  {
	  // Example implementation:
	  uint32_t i, tmp;
	  for(i = 0; i < size; i = i + 4)
	  {
		  tmp = ((uint32_t)buffer[i] << 24)
			  | ((uint32_t)buffer[i+1] << 16)
			  | ((uint32_t)buffer[i+2] <<  8)
			  |  (uint32_t)buffer[i+3];
		  
		  memcpy(&(buffer[i]), &tmp, 4);
	  }
  }
  
  /**
   * Wait in milliseconds
   */
  uint8_t VL53L8CX_WaitMs(VL53L8CX_Platform * p_platform, uint32_t time_ms)
  {
	  (void)p_platform;  // avoid unused warning
	  usleep(time_ms * 1000);
	  return 0;
  }
  
  /**
   * Wait for data ready, either by polling or kernel interrupt
   */
  uint8_t VL53L8CX_wait_for_dataready(VL53L8CX_Platform *p_platform)
  {
  #ifdef STMVL53L8CX_KERNEL
	  if (ioctl(p_platform->fd, ST_TOF_IOCTL_WAIT_FOR_INTERRUPT) < 0)
		  return 0;
  #else
	  // Polling approach
	  VL53L8CX_Configuration * p_dev = 
		  (VL53L8CX_Configuration *)(p_platform - offsetof(VL53L8CX_Configuration, platform));
	  
	  uint8_t isReady = 0;
	  do {
		  VL53L8CX_WaitMs(p_platform, 5);
		  vl53l8cx_check_data_ready(p_dev, &isReady);
	  } while (isReady == 0);
  #endif
	  return 1;
  }
  