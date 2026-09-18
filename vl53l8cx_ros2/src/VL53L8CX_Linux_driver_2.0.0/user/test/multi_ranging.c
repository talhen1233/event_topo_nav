/*******************************************************************************
Copyright (C) 2022, STMicroelectronics International N.V.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of STMicroelectronics nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS ARE DISCLAIMED.
IN NO EVENT SHALL STMICROELECTRONICS INTERNATIONAL N.V. BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************************/

#include <unistd.h>
#include <signal.h>
#include <dlfcn.h>

#include <stdio.h>
#include <string.h>

#include "vl53l8cx_api.h"

#include "examples.h"

int exit_main_loop = 0;

void sighandler(int signal)
{
	printf("SIGNAL Handler called, signal = %d\n", signal);
	exit_main_loop  = 1;
}

int main(int argc, char ** argv)
{
	#define NB_OF_DEV 4
	uint8_t i;
	char choice[20];
	int status;
	VL53L8CX_Configuration 	Dev[NB_OF_DEV];

	/*********************************/
	/*   Power on sensor and init    */
	/*********************************/

	memset(&Dev, 0, sizeof(Dev));
	Dev[0].platform.spi_num = 0;
	Dev[0].platform.spi_cs = 0;

	Dev[1].platform.spi_num = 0;
	Dev[1].platform.spi_cs = 1;

	Dev[2].platform.spi_num = 1;
	Dev[2].platform.spi_cs = 0;

	Dev[3].platform.spi_num = 1;
	Dev[3].platform.spi_cs = 2;

	for (i=0; i< NB_OF_DEV; i++)
	{
		status = vl53l8cx_comms_init(&Dev[i].platform);
		if(status)
		{
			printf("VL53L8CX comms init failed on spidev%d.%d\n", Dev[i].platform.spi_num, Dev[i].platform.spi_cs);
			return -1;
		}
	}

	printf("Starting dual ranging with ULD version %s\n", VL53L8CX_API_REVISION);

	do {
		printf("----------------------------------------------------------------------------------------------------------\n");
		printf(" VL53L8CX uld driver test dual devices example menu \n");
		printf(" ------------------ Ranging menu ------------------\n");
		printf(" 1 : first device #0 only ranging \n");
		printf(" 2 : second device #1 only ranging\n");
		printf(" 3 : third device #2 only ranging\n");
		printf(" 4 : fourth device #3 only ranging\n");
		printf(" 5 : both devices ranging\n");
		printf(" 6 : exit\n");
		printf("----------------------------------------------------------------------------------------------------------\n");

		printf("Your choice ?\n ");
		scanf("%s", choice);

		if (strcmp(choice, "1") == 0) {
			printf("Starting Test device #0\n");
			status = example1(&Dev[0]);
			printf("\n");
		}
		else if (strcmp(choice, "2") == 0) {
			printf("Starting Test device #1\n");
			status = example1(&Dev[1]);
			printf("\n");
		}
		else if (strcmp(choice, "3") == 0) {
			printf("Starting Test device #2\n");
			status = example1(&Dev[2]);
			printf("\n");
		}
		else if (strcmp(choice, "4") == 0) {
			printf("Starting Test device #3\n");
			status = example1(&Dev[3]);
			printf("\n");
		}
		else if (strcmp(choice, "5") == 0) {
			printf("Starting Test 5\n");
			status = example_multi(Dev, NB_OF_DEV);
			printf("\n");
		}
		
		else if (strcmp(choice, "6") == 0){
			exit_main_loop = 1;
		}
		
		else{
			printf("Invalid choice\n");
		}

	} while (!exit_main_loop);

	for (i=0; i< NB_OF_DEV; i++)
	{
		vl53l8cx_comms_close(&Dev[i].platform);
	}

	return 0;
}
