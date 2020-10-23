#include "gslib.h"
#include <occa.hpp>
#include "ogs.hpp"
#include "ogsKernels.hpp"
#include "ogsInterface.h"

enum ogs_mode { OGS_DEFAULT, OGS_HOSTMPI, OGS_DEVICEMPI };

void mygsStart(occa::memory o_v, const char* type, const char* op, ogs_t* ogs, ogs_mode ogs_mode);
void mygsFinish(occa::memory o_v, const char* type, const char* op, ogs_t* ogs, ogs_mode ogs_mode);
void mygsSetup(ogs_t* ogs);
void mygsEnableTimer(int timers);
