static inline void nrLDPC_bnProc_BG1_R23_128(int8_t* bnProcBuf,int8_t* bnProcBufRes,  int8_t* llrRes, uint16_t Z ) {
        uint32_t M, i; 
// Process group with 2 CNs 
 M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[216 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[216 + i ], ((simde__m128i*) bnProcBuf)[216 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[240 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[216 + i ], ((simde__m128i*) bnProcBuf)[240 + i]);
}
// Process group with 3 CNs 
       M = (5*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[264 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[240 + i ], ((simde__m128i*) bnProcBuf)[264 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[384 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[240 + i ], ((simde__m128i*) bnProcBuf)[384 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[504 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[240 + i ], ((simde__m128i*) bnProcBuf)[504 + i]);
}
// Process group with 4 CNs 
       M = (3*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[624 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[360 + i ], ((simde__m128i*) bnProcBuf)[624 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[696 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[360 + i ], ((simde__m128i*) bnProcBuf)[696 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[768 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[360 + i ], ((simde__m128i*) bnProcBuf)[768 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[840 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[360 + i ], ((simde__m128i*) bnProcBuf)[840 + i]);
}
// Process group with 5 CNs 
       M = (7*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[912 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[432 + i ], ((simde__m128i*) bnProcBuf)[912 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1080 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[432 + i ], ((simde__m128i*) bnProcBuf)[1080 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1248 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[432 + i ], ((simde__m128i*) bnProcBuf)[1248 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1416 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[432 + i ], ((simde__m128i*) bnProcBuf)[1416 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1584 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[432 + i ], ((simde__m128i*) bnProcBuf)[1584 + i]);
}
// Process group with 6 CNs 
       M = (8*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1752 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[600 + i ], ((simde__m128i*) bnProcBuf)[1752 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1944 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[600 + i ], ((simde__m128i*) bnProcBuf)[1944 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2136 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[600 + i ], ((simde__m128i*) bnProcBuf)[2136 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2328 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[600 + i ], ((simde__m128i*) bnProcBuf)[2328 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2520 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[600 + i ], ((simde__m128i*) bnProcBuf)[2520 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2712 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[600 + i ], ((simde__m128i*) bnProcBuf)[2712 + i]);
}
// Process group with 7 CNs 
// Process group with 8 CNs 
// Process group with 9 CNs 
// Process group with 10 CNs 
// Process group with 11 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2904 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[792 + i ], ((simde__m128i*) bnProcBuf)[2904 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2928 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[792 + i ], ((simde__m128i*) bnProcBuf)[2928 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2952 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[792 + i ], ((simde__m128i*) bnProcBuf)[2952 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2976 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[792 + i ], ((simde__m128i*) bnProcBuf)[2976 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3000 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[792 + i ], ((simde__m128i*) bnProcBuf)[3000 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3024 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[792 + i ], ((simde__m128i*) bnProcBuf)[3024 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3048 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[792 + i ], ((simde__m128i*) bnProcBuf)[3048 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3072 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[792 + i ], ((simde__m128i*) bnProcBuf)[3072 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3096 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[792 + i ], ((simde__m128i*) bnProcBuf)[3096 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3120 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[792 + i ], ((simde__m128i*) bnProcBuf)[3120 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3144 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[792 + i ], ((simde__m128i*) bnProcBuf)[3144 + i]);
}
// Process group with 12 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3168 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[816 + i ], ((simde__m128i*) bnProcBuf)[3168 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3192 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[816 + i ], ((simde__m128i*) bnProcBuf)[3192 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3216 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[816 + i ], ((simde__m128i*) bnProcBuf)[3216 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3240 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[816 + i ], ((simde__m128i*) bnProcBuf)[3240 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3264 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[816 + i ], ((simde__m128i*) bnProcBuf)[3264 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3288 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[816 + i ], ((simde__m128i*) bnProcBuf)[3288 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3312 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[816 + i ], ((simde__m128i*) bnProcBuf)[3312 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3336 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[816 + i ], ((simde__m128i*) bnProcBuf)[3336 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3360 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[816 + i ], ((simde__m128i*) bnProcBuf)[3360 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3384 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[816 + i ], ((simde__m128i*) bnProcBuf)[3384 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3408 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[816 + i ], ((simde__m128i*) bnProcBuf)[3408 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3432 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[816 + i ], ((simde__m128i*) bnProcBuf)[3432 + i]);
}
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
