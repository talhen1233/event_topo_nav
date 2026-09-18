/*******************************************************************************
* Copyright (c) 2022, STMicroelectronics
* All rights reserved.
*
* This code is an adaptation to run the step 2 example indefinitely and stop
* upon receiving a "stop" command from stdin.
*******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/select.h>

#include "vl53l8cx_api.h"

// Include this if needed for VL53L8CX_WaitMs or provide your own delay method.
#include "platform.h" 

// Helper function to check for 'stop' command from stdin
static int check_for_stop_command(void) {
    fd_set readfds;
    struct timeval tv;
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);

    // No timeout, just poll
    tv.tv_sec = 0;
    tv.tv_usec = 0;

    int ret = select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv);
    if (ret > 0 && FD_ISSET(STDIN_FILENO, &readfds)) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), stdin) != NULL) {
            if (strstr(buffer, "stop") != NULL) {
                return 1; // Stop command received
            }
        }
    }
    return 0;
}

int example2(VL53L8CX_Configuration *p_dev)
{
    uint8_t status, isAlive, isReady, i;
    uint32_t integration_time_ms;
    VL53L8CX_ResultsData Results; // Results data from VL53L8CX

    // Check sensor presence
    status = vl53l8cx_is_alive(p_dev, &isAlive);
    if(!isAlive || status)
    {
        printf("VL53L8CX not detected at requested address\n");
        return status;
    }

    // Init sensor
    status = vl53l8cx_init(p_dev);
    if(status)
    {
        printf("VL53L8CX ULD Loading failed\n");
        return status;
    }

    printf("VL53L8CX ULD ready ! (Version : %s)\n", VL53L8CX_API_REVISION);

    // Set resolution to 8x8
    status = vl53l8cx_set_resolution(p_dev, VL53L8CX_RESOLUTION_8X8);
    if(status)
    {
        printf("vl53l8cx_set_resolution failed, status %u\n", status);
        return status;
    }

    // Set ranging frequency to 10Hz
    status = vl53l8cx_set_ranging_frequency_hz(p_dev, 30);
    if(status)
    {
        printf("vl53l8cx_set_ranging_frequency_hz failed, status %u\n", status);
        return status;
    }

    // Set target order to closest
    status = vl53l8cx_set_target_order(p_dev, VL53L8CX_TARGET_ORDER_CLOSEST);
    if(status)
    {
        printf("vl53l8cx_set_target_order failed, status %u\n", status);
        return status;
    }

    // Get current integration time
    status = vl53l8cx_get_integration_time_ms(p_dev, &integration_time_ms);
    if(status)
    {
        printf("vl53l8cx_get_integration_time_ms failed, status %u\n", status);
        return status;
    }
    printf("Current integration time is : %d ms\n", integration_time_ms);

    // Start ranging
    status = vl53l8cx_start_ranging(p_dev);
    if(status)
    {
        printf("vl53l8cx_start_ranging failed, status %u\n", status);
        return status;
    }

    // Infinite loop until "stop" command is received
    while (1)
    {
        status = vl53l8cx_check_data_ready(p_dev, &isReady);
        if (isReady)
        {
            vl53l8cx_get_ranging_data(p_dev, &Results);

            // Print data for all 64 zones
            printf("Print data no : %3u\n", p_dev->streamcount);
            for(i = 0; i < 64; i++)
            {
                printf("Zone : %3d, Status : %3u, Distance : %4d mm\n",
                       i,
                       Results.target_status[VL53L8CX_NB_TARGET_PER_ZONE*i],
                       Results.distance_mm[VL53L8CX_NB_TARGET_PER_ZONE*i]);
            }
            printf("\n");
        }

        // Check if we received a stop command from stdin
        if (check_for_stop_command()) {
            printf("Stop command received. Ending loop.\n");
            break;
        }

        // Small delay to reduce CPU usage
        VL53L8CX_WaitMs(&p_dev->platform, 5);
    }

    status = vl53l8cx_stop_ranging(p_dev);
    printf("End of ULD demo\n");
    return status;
}
