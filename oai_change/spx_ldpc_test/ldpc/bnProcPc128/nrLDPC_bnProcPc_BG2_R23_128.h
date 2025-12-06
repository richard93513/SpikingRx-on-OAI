#include <stdint.h>
#include "PHY/sse_intrin.h"
static inline void nrLDPC_bnProcPc_BG2_R23_128(int8_t* bnProcBuf,int8_t* bnProcBufRes,int8_t* llrRes ,  int8_t* llrProcBuf, uint16_t Z  ) {
  // Process group with 1 CN
        uint32_t M = (3*Z + 15)>>4;
        simde__m128i* p_bnProcBuf    = (simde__m128i*) &bnProcBuf    [0];
        simde__m128i* p_bnProcBufRes = (simde__m128i*) &bnProcBufRes [0];
        simde__m128i* p_llrProcBuf   = (simde__m128i*) &llrProcBuf   [0];
        simde__m128i* p_llrRes       = (simde__m128i*) &llrRes       [0];
        simde__m128i ymm0, ymm1, ymmRes0, ymmRes1;
        for (int i=0;i<M;i++) {
          p_bnProcBufRes[i] = p_llrProcBuf[i];
          ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf [i]);
          ymm1 = simde_mm_cvtepi8_epi16(p_llrProcBuf[i]);
          ymmRes0 = simde_mm_adds_epi16(ymm0, ymm1);
          ymm0 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf [i],8));
          ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_llrProcBuf[i],8));
          ymmRes1 = simde_mm_adds_epi16(ymm0, ymm1);
          *p_llrRes = simde_mm_packs_epi16(ymmRes0, ymmRes1);
          p_llrRes++;
        }
  M = (3*Z + 15)>>4;
  p_bnProcBuf     = (simde__m128i*) &bnProcBuf    [1152];
  p_llrProcBuf    = (simde__m128i*) &llrProcBuf   [1152];
  p_llrRes        = (simde__m128i*) &llrRes       [1152];
        for (int i=0;i<M;i++) {
        ymmRes0 = simde_mm_cvtepi8_epi16(p_bnProcBuf [i]);
        ymmRes1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf [i],8));
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[72 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[72 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0    = simde_mm_cvtepi8_epi16(p_llrProcBuf[i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1    = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_llrProcBuf[i],8));
        ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1);
        *p_llrRes = simde_mm_packs_epi16(ymmRes0, ymmRes1);
        p_llrRes++;
   }
  M = (5*Z + 15)>>4;
  p_bnProcBuf     = (simde__m128i*) &bnProcBuf    [3456];
  p_llrProcBuf    = (simde__m128i*) &llrProcBuf   [2304];
  p_llrRes        = (simde__m128i*) &llrRes       [2304];
        for (int i=0;i<M;i++) {
        ymmRes0 = simde_mm_cvtepi8_epi16(p_bnProcBuf [i]);
        ymmRes1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf [i],8));
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[120 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[120 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[240 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[240 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0    = simde_mm_cvtepi8_epi16(p_llrProcBuf[i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1    = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_llrProcBuf[i],8));
        ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1);
        *p_llrRes = simde_mm_packs_epi16(ymmRes0, ymmRes1);
        p_llrRes++;
   }
  M = (3*Z + 15)>>4;
  p_bnProcBuf     = (simde__m128i*) &bnProcBuf    [9216];
  p_llrProcBuf    = (simde__m128i*) &llrProcBuf   [4224];
  p_llrRes        = (simde__m128i*) &llrRes       [4224];
        for (int i=0;i<M;i++) {
        ymmRes0 = simde_mm_cvtepi8_epi16(p_bnProcBuf [i]);
        ymmRes1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf [i],8));
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[72 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[72 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[144 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[144 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[216 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[216 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0    = simde_mm_cvtepi8_epi16(p_llrProcBuf[i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1    = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_llrProcBuf[i],8));
        ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1);
        *p_llrRes = simde_mm_packs_epi16(ymmRes0, ymmRes1);
        p_llrRes++;
   }
  M = (2*Z + 15)>>4;
  p_bnProcBuf     = (simde__m128i*) &bnProcBuf    [13824];
  p_llrProcBuf    = (simde__m128i*) &llrProcBuf   [5376];
  p_llrRes        = (simde__m128i*) &llrRes       [5376];
        for (int i=0;i<M;i++) {
        ymmRes0 = simde_mm_cvtepi8_epi16(p_bnProcBuf [i]);
        ymmRes1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf [i],8));
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[48 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[48 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[96 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[96 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[144 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[144 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[192 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[192 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0    = simde_mm_cvtepi8_epi16(p_llrProcBuf[i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1    = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_llrProcBuf[i],8));
        ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1);
        *p_llrRes = simde_mm_packs_epi16(ymmRes0, ymmRes1);
        p_llrRes++;
   }
  M = (1*Z + 15)>>4;
  p_bnProcBuf     = (simde__m128i*) &bnProcBuf    [17664];
  p_llrProcBuf    = (simde__m128i*) &llrProcBuf   [6144];
  p_llrRes        = (simde__m128i*) &llrRes       [6144];
        for (int i=0;i<M;i++) {
        ymmRes0 = simde_mm_cvtepi8_epi16(p_bnProcBuf [i]);
        ymmRes1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf [i],8));
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[24 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[24 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[48 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[48 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[72 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[72 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[96 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[96 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0 = simde_mm_cvtepi8_epi16(p_bnProcBuf[120 + i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1 = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_bnProcBuf[120 + i],8));
       ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1); 
        ymm0    = simde_mm_cvtepi8_epi16(p_llrProcBuf[i]);
        ymmRes0 = simde_mm_adds_epi16(ymmRes0, ymm0);
        ymm1    = simde_mm_cvtepi8_epi16(simde_mm_srli_si128(p_llrProcBuf[i],8));
        ymmRes1 = simde_mm_adds_epi16(ymmRes1, ymm1);
        *p_llrRes = simde_mm_packs_epi16(ymmRes0, ymmRes1);
        p_llrRes++;
   }
}
