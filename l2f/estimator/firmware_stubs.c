// Firmware services that the compiled estimator files reference but the simulator does not
// use (memory interface, OOTX decoding of the base station data stream, log handlers, the
// microsecond clock of the pulse decoder). A firmware ASSERT that fails marks the running
// estimator failed (cfEstimatorFailed); the simulator ends the episode as a crash, as the
// assert would halt the drone.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "cf_estimator.h"
#include "mem.h"
#include "ootx_decoder.h"
#include "pulse_processor.h"

pulseProcessor_t lighthouseCoreState;  // lighthouse_core.c global, used only by the memory handlers

void assertFail(char* exp, char* file, int line) { cfEstimatorAssertFailed(exp, file, line); }

void memoryRegisterHandler(const MemoryHandlerDef_t* handlerDef) { (void)handlerDef; }

bool ootxDecoderProcessBit(ootxDecoderState_t* state, int data) {
  (void)state;
  (void)data;
  return false;
}

float statsCntRateLogHandler(uint32_t timestamp, void* data) {
  (void)timestamp;
  (void)data;
  return 0;
}

uint64_t usecTimestamp(void) { return 0; }
