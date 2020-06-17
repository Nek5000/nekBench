#include "gslib.h"
#include <occa.hpp>
#include "ogs.hpp"
#include "ogsKernels.hpp"
#include "ogsInterface.h"

#ifdef __cplusplus
extern "C" {
#endif

void gsStart(occa::memory o_v, const char *type, const char *op, ogs_t *ogs);
void gsFinish(occa::memory o_v, const char *type, const char *op, ogs_t *ogs);

#ifdef __cplusplus
}
#endif
