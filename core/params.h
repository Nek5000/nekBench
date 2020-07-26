/* offsets for geometric factors */
#define RXID 0
#define RYID 1
#define SXID 2
#define SYID 3
#define  JID 4
#define JWID 5
#define IJWID 6
#define RZID 7
#define SZID 8
#define TXID 9
#define TYID 10
#define TZID 11

/* offsets for second order geometric factors */
#if 0
#define G00ID 0
#define G01ID 1
#define G11ID 2
#define GWJID 3
#define G12ID 4
#define G02ID 5
#define G22ID 6
#else
#define p_GWJID 0
#define p_G00ID 1
#define p_G01ID 2
#define p_G02ID 3
#define p_G11ID 4
#define p_G12ID 5
#define p_G22ID 6

#define GWJID 0
#define G00ID 1
#define G01ID 2
#define G02ID 3
#define G11ID 4
#define G12ID 5
#define G22ID 6
#endif

/* offsets for nx, ny, sJ, 1/J */
#define NXID 0
#define NYID 1
#define SJID 2
#define IJID 3
#define IHID 4
#define WSJID 5
#define WIJID 6
#define NZID 7
#define SURXID 8
#define SURYID 9
#define SURZID 10

#define p_GWJID 0
#define p_G00ID 1
#define p_G01ID 2
#define p_G02ID 3
#define p_G11ID 4
#define p_G12ID 5
#define p_G22ID 6
#define p_Nggeo 7
#define p_Nvgeo 12
