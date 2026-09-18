/*******************************************************************************
* Copyright (c) 2022, STMicroelectronics - All Rights Reserved
*
* This file is part of the VL53L8CX Ultra Lite Driver and is dual licensed,
* either 'STMicroelectronics Proprietary license'
* or 'BSD 3-clause "New" or "Revised" License' , at your option.
*
********************************************************************************
*
* 'STMicroelectronics Proprietary license'
*
********************************************************************************
*
* License terms: STMicroelectronics Proprietary in accordance with licensing
* terms at www.st.com/sla0081
*
* STMicroelectronics confidential
* Reproduction and Communication of this document is strictly prohibited unless
* specifically authorized in writing by STMicroelectronics.
*
*
********************************************************************************
*
* Alternatively, the VL53L8CX Ultra Lite Driver may be distributed under the
* terms of 'BSD 3-clause "New" or "Revised" License', in which case the
* following provisions apply instead of the ones mentioned above :
*
********************************************************************************
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

/***********************************/
/*   VL53L8CX ULD basic example    */
/***********************************/
/*
* This example is the most basic. It initializes the VL53L8CX ULD, and starts
* a ranging to capture 10 frames.
*
* By default, ULD is configured to have the following settings :
* - Resolution 4x4
* - Ranging period 1Hz
*
* In this example, we also suppose that the number of target per zone is
* set to 1 , and all output are enabled (see file platform.h).
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "vl53l8cx_api.h"

int example_dual(VL53L8CX_Configuration *p_dev1, VL53L8CX_Configuration *p_dev2)
{

	/*********************************/
	/*   VL53L8CX ranging variables  */
	/*********************************/

	uint8_t 				status, loop, isAlive, i;
	uint8_t					isReady1, isReady2;
	VL53L8CX_ResultsData 	Results1, Results2;		/* Results data from VL53L8CX */


	/*********************************/
	/*   Power on sensor and init    */
	/*********************************/

	/* (Optional) Check if there is a VL53L8CX sensor connected */
	status = vl53l8cx_is_alive(p_dev1, &isAlive);
	if(!isAlive || status)
	{
		printf("First VL53L8CX not detected at requested address\n");
		return status;
	}

	status = vl53l8cx_is_alive(p_dev2, &isAlive);
	if(!isAlive || status)
	{
		printf("Second VL53L8CX not detected at requested address\n");
		return status;
	}

	/* (Mandatory) Init VL53L8CX sensor */
	status = vl53l8cx_init(p_dev1);
	if(status)
	{
		printf("First VL53L8CX ULD Loading failed\n");
		return status;
	}

	VL53L8CX_WaitMs(&p_dev1->platform, 2);

	status = vl53l8cx_init(p_dev2);
	if(status)
	{
		printf("Second VL53L8CX ULD Loading failed\n");
		return status;
	}

	printf("VL53L8CX ULD ready ! (Version : %s)\n",
			VL53L8CX_API_REVISION);

/*
	status = vl53l8cx_set_resolution(p_dev1, VL53L8CX_RESOLUTION_4X4);
    status |= vl53l8cx_set_ranging_mode(p_dev1, VL53L8CX_RANGING_MODE_CONTINUOUS);
    status |= vl53l8cx_set_integration_time_ms(p_dev1, 100);
    status |= vl53l8cx_set_ranging_frequency_hz(p_dev1, 5);

	status = vl53l8cx_set_resolution(p_dev2, VL53L8CX_RESOLUTION_4X4);
    status |= vl53l8cx_set_ranging_mode(p_dev2, VL53L8CX_RANGING_MODE_CONTINUOUS);
    status |= vl53l8cx_set_integration_time_ms(p_dev2, 100);
    status |= vl53l8cx_set_ranging_frequency_hz(p_dev2, 5);
*/

	/*********************************/
	/*         Ranging loop          */
	/*********************************/

	status = vl53l8cx_start_ranging(p_dev1);
	VL53L8CX_WaitMs(&p_dev1->platform, 2);
	status = vl53l8cx_start_ranging(p_dev2);

	loop = 0;
	while(loop < 20)
	{
		/* Use polling function to know when a new measurement is ready */
 
		status = vl53l8cx_check_data_ready(p_dev1, &isReady1);

		status = vl53l8cx_check_data_ready(p_dev2, &isReady2);

		if(isReady1)
		{
			vl53l8cx_get_ranging_data(p_dev1, &Results1);
		}

		if(isReady2)
		{
			vl53l8cx_get_ranging_data(p_dev2, &Results2);
		}
		
		if (isReady1)
		{
			/* As the sensor is set in 4x4 mode by default, we have a total 
			 * of 16 zones to print. For this example, only the data of first zone are 
			 * print */
			printf("satel 1 Print data no : %3u \n", p_dev1->streamcount);
			for(i = 0; i < 16; i++)
			{
				printf("Zone : %3d, Status : %3u, Distance : %4d mm\n",
					i,
					Results1.target_status[VL53L8CX_NB_TARGET_PER_ZONE*i],
					Results1.distance_mm[VL53L8CX_NB_TARGET_PER_ZONE*i]);
			}
			printf("\n");
			loop++;
		}

		if (isReady2)
		{
			/* As the sensor is set in 4x4 mode by default, we have a total 
			 * of 16 zones to print. For this example, only the data of first zone are 
			 * print */
			printf("satel 2 Print data no : %3u \n", p_dev2->streamcount);
			for(i = 0; i < 16; i++)
			{
				printf("Zone : %3d, Status : %3u, Distance : %4d mm\n",
					i,
					Results2.target_status[VL53L8CX_NB_TARGET_PER_ZONE*i],
					Results2.distance_mm[VL53L8CX_NB_TARGET_PER_ZONE*i]);
			}
			printf("\n");
			loop++;
		}

		/* Wait a few ms to avoid too high polling (function in platform file, not in API) */
		VL53L8CX_WaitMs(&p_dev1->platform, 2);
	}
	status = vl53l8cx_stop_ranging(p_dev1);
	status = vl53l8cx_stop_ranging(p_dev2);
	printf("End of ULD demo\n");
	return status;
}

int example_multi(VL53L8CX_Configuration tdev[], uint8_t max_dev)
{
	/*********************************/
	/*   VL53L8CX ranging variables  */
	/*********************************/

	uint8_t 				status, loop, isAlive, i, idev;
	uint8_t					isReady;
	VL53L8CX_ResultsData 	Results;


	/*********************************/
	/*   Power on sensor and init    */
	/*********************************/
	status = 0;
	for (idev = 0; idev < max_dev; idev++) {
		
		status = vl53l8cx_is_alive(&tdev[idev], &isAlive);
		if(!isAlive || status)
		{
			printf("VL53L8CX #%d not detected at requested address \n",idev);
			return status;
		}
		status = vl53l8cx_init(&tdev[idev]);
		if(status)
		{
			printf("VL53L8CX #%d ULD Loading failed\n", idev);
			return status;
		}

		status = vl53l8cx_set_resolution(&tdev[idev], VL53L8CX_RESOLUTION_4X4);
		status |= vl53l8cx_set_ranging_mode(&tdev[idev], VL53L8CX_RANGING_MODE_CONTINUOUS);
		status |= vl53l8cx_set_integration_time_ms(&tdev[idev], 30);
		status |= vl53l8cx_set_ranging_frequency_hz(&tdev[idev], 25);

		if(!status)
			printf("VL53L8CX #%d initialized\n",idev);

	}

	printf("VL53L8CX ULD ready (Version : %s)\n",
			VL53L8CX_API_REVISION);

	for (idev = 0; idev < max_dev; idev++) {
		status = vl53l8cx_start_ranging(&tdev[idev]);
		if(!status)
			printf("VL53L8CX #%d started\n",idev);
	}

	/*********************************/
	/*         Ranging loop          */
	/*********************************/

	loop = 0;
	while(loop < (10 * max_dev))
	{
		/* Use polling function to know when a new measurement is ready */
 
		for (idev = 0; idev < max_dev; idev++) {
		
			status = vl53l8cx_check_data_ready(&tdev[idev], &isReady);

			if(isReady)
			{
				vl53l8cx_get_ranging_data(&tdev[idev], &Results);
			}
			
			if (isReady)
			{
				/* As the sensor is set in 4x4 mode by default, we have a total 
				 * of 16 zones to print. For this example, only the data of first zone are 
				 * print */
				printf("satel #%d Print data no : %3u \n", idev, (&tdev[idev])->streamcount);
				for(i = 0; i < 16; i++)
				{
					printf("Zone : %3d, Status : %3u, Distance : %4d mm\n",
						i,
						Results.target_status[VL53L8CX_NB_TARGET_PER_ZONE*i],
						Results.distance_mm[VL53L8CX_NB_TARGET_PER_ZONE*i]);
				}
				printf("\n");
				loop++;
			}
			VL53L8CX_WaitMs(NULL, 2);
		}		
	}

	for (idev = 0; idev < max_dev; idev++)
		status = vl53l8cx_stop_ranging(&tdev[idev]);

	printf("End of ULD demo\n");
	return status;
}
