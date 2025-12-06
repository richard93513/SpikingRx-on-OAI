static inline void nrLDPC_bnProc_BG2_R23_128(int8_t* bnProcBuf,int8_t* bnProcBufRes,  int8_t* llrRes, uint16_t Z  ) {
        uint32_t M, i; 
// Process group with 2 CNs 
 M = (3*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[72 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[72 + i ], ((simde__m128i*) bnProcBuf)[72 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[144 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[72 + i ], ((simde__m128i*) bnProcBuf)[144 + i]);
}
// Process group with 3 CNs 
       M = (5*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[216 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[144 + i ], ((simde__m128i*) bnProcBuf)[216 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[336 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[144 + i ], ((simde__m128i*) bnProcBuf)[336 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[456 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[144 + i ], ((simde__m128i*) bnProcBuf)[456 + i]);
}
// Process group with 4 CNs 
       M = (3*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[576 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[264 + i ], ((simde__m128i*) bnProcBuf)[576 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[648 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[264 + i ], ((simde__m128i*) bnProcBuf)[648 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[720 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[264 + i ], ((simde__m128i*) bnProcBuf)[720 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[792 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[264 + i ], ((simde__m128i*) bnProcBuf)[792 + i]);
}
// Process group with 5 CNs 
       M = (2*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[864 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[336 + i ], ((simde__m128i*) bnProcBuf)[864 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[912 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[336 + i ], ((simde__m128i*) bnProcBuf)[912 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[960 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[336 + i ], ((simde__m128i*) bnProcBuf)[960 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1008 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[336 + i ], ((simde__m128i*) bnProcBuf)[1008 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1056 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[336 + i ], ((simde__m128i*) bnProcBuf)[1056 + i]);
}
// Process group with 6 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1104 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[384 + i ], ((simde__m128i*) bnProcBuf)[1104 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1128 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[384 + i ], ((simde__m128i*) bnProcBuf)[1128 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1152 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[384 + i ], ((simde__m128i*) bnProcBuf)[1152 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1176 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[384 + i ], ((simde__m128i*) bnProcBuf)[1176 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1200 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[384 + i ], ((simde__m128i*) bnProcBuf)[1200 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1224 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[384 + i ], ((simde__m128i*) bnProcBuf)[1224 + i]);
}
// Process group with 7 CNs 
// Process group with 8 CNs 
// Process group with 9 CNs 
// Process group with 10 CNs 
// Process group with 11 CNs 
// Process group with 12 CNs 
// Process group with 13 CNs 
// Process group with 14 CNs 
// Process group with 15 CNs 
// Process group with 16 CNs 
// Process group with 17 CNs 
// Process group with 18 CNs 
// Process group with 19 CNs 
// Process group with 20 CNs 
// Process group with 21 CNs 
// Process group with 22 CNs 
// Process group with <23 CNs 
// Process group with 24 CNs 
// Process group with 25 CNs 
// Process group with 26 CNs 
// Process group with 27 CNs 
// Process group with 28 CNs 
// Process group with 29 CNs 
// Process group with 30 CNs 
}
