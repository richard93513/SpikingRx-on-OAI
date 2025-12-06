static inline void nrLDPC_bnProc_BG2_R15_128(int8_t* bnProcBuf,int8_t* bnProcBufRes,  int8_t* llrRes, uint16_t Z  ) {
        uint32_t M, i; 
// Process group with 2 CNs 
// Process group with 3 CNs 
// Process group with 4 CNs 
// Process group with 5 CNs 
       M = (2*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[912 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[912 + i ], ((simde__m128i*) bnProcBuf)[912 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[960 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[912 + i ], ((simde__m128i*) bnProcBuf)[960 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1008 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[912 + i ], ((simde__m128i*) bnProcBuf)[1008 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1056 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[912 + i ], ((simde__m128i*) bnProcBuf)[1056 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1104 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[912 + i ], ((simde__m128i*) bnProcBuf)[1104 + i]);
}
// Process group with 6 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1152 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[960 + i ], ((simde__m128i*) bnProcBuf)[1152 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1176 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[960 + i ], ((simde__m128i*) bnProcBuf)[1176 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1200 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[960 + i ], ((simde__m128i*) bnProcBuf)[1200 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1224 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[960 + i ], ((simde__m128i*) bnProcBuf)[1224 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1248 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[960 + i ], ((simde__m128i*) bnProcBuf)[1248 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1272 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[960 + i ], ((simde__m128i*) bnProcBuf)[1272 + i]);
}
// Process group with 7 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1296 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[984 + i ], ((simde__m128i*) bnProcBuf)[1296 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1320 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[984 + i ], ((simde__m128i*) bnProcBuf)[1320 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1344 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[984 + i ], ((simde__m128i*) bnProcBuf)[1344 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1368 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[984 + i ], ((simde__m128i*) bnProcBuf)[1368 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1392 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[984 + i ], ((simde__m128i*) bnProcBuf)[1392 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1416 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[984 + i ], ((simde__m128i*) bnProcBuf)[1416 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1440 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[984 + i ], ((simde__m128i*) bnProcBuf)[1440 + i]);
}
// Process group with 8 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1464 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1008 + i ], ((simde__m128i*) bnProcBuf)[1464 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1488 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1008 + i ], ((simde__m128i*) bnProcBuf)[1488 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1512 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1008 + i ], ((simde__m128i*) bnProcBuf)[1512 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1536 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1008 + i ], ((simde__m128i*) bnProcBuf)[1536 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1560 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1008 + i ], ((simde__m128i*) bnProcBuf)[1560 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1584 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1008 + i ], ((simde__m128i*) bnProcBuf)[1584 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1608 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1008 + i ], ((simde__m128i*) bnProcBuf)[1608 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1632 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1008 + i ], ((simde__m128i*) bnProcBuf)[1632 + i]);
}
// Process group with 9 CNs 
       M = (2*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1656 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[1656 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1704 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[1704 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1752 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[1752 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1800 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[1800 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1848 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[1848 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1896 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[1896 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1944 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[1944 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1992 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[1992 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2040 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[2040 + i]);
}
// Process group with 10 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2088 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1080 + i ], ((simde__m128i*) bnProcBuf)[2088 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2112 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1080 + i ], ((simde__m128i*) bnProcBuf)[2112 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2136 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1080 + i ], ((simde__m128i*) bnProcBuf)[2136 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2160 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1080 + i ], ((simde__m128i*) bnProcBuf)[2160 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2184 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1080 + i ], ((simde__m128i*) bnProcBuf)[2184 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2208 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1080 + i ], ((simde__m128i*) bnProcBuf)[2208 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2232 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1080 + i ], ((simde__m128i*) bnProcBuf)[2232 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2256 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1080 + i ], ((simde__m128i*) bnProcBuf)[2256 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2280 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1080 + i ], ((simde__m128i*) bnProcBuf)[2280 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2304 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1080 + i ], ((simde__m128i*) bnProcBuf)[2304 + i]);
}
// Process group with 11 CNs 
// Process group with 12 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2328 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[2328 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2352 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[2352 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2376 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[2376 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2400 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[2400 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2424 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[2424 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2448 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[2448 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2472 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[2472 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2496 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[2496 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2520 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[2520 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2544 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[2544 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2568 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[2568 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2592 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[2592 + i]);
}
// Process group with 13 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2616 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1128 + i ], ((simde__m128i*) bnProcBuf)[2616 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2640 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1128 + i ], ((simde__m128i*) bnProcBuf)[2640 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2664 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1128 + i ], ((simde__m128i*) bnProcBuf)[2664 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2688 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1128 + i ], ((simde__m128i*) bnProcBuf)[2688 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2712 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1128 + i ], ((simde__m128i*) bnProcBuf)[2712 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2736 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1128 + i ], ((simde__m128i*) bnProcBuf)[2736 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2760 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1128 + i ], ((simde__m128i*) bnProcBuf)[2760 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2784 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1128 + i ], ((simde__m128i*) bnProcBuf)[2784 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2808 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1128 + i ], ((simde__m128i*) bnProcBuf)[2808 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2832 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1128 + i ], ((simde__m128i*) bnProcBuf)[2832 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2856 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1128 + i ], ((simde__m128i*) bnProcBuf)[2856 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2880 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1128 + i ], ((simde__m128i*) bnProcBuf)[2880 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2904 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1128 + i ], ((simde__m128i*) bnProcBuf)[2904 + i]);
}
// Process group with 14 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2928 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[2928 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2952 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[2952 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2976 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[2976 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3000 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[3000 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3024 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[3024 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3048 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[3048 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3072 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[3072 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3096 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[3096 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3120 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[3120 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3144 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[3144 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3168 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[3168 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3192 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[3192 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3216 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[3216 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3240 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1152 + i ], ((simde__m128i*) bnProcBuf)[3240 + i]);
}
// Process group with 15 CNs 
// Process group with 16 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3264 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3264 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3288 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3288 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3312 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3312 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3336 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3336 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3360 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3360 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3384 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3384 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3408 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3408 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3432 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3432 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3456 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3456 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3480 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3480 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3504 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3504 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3528 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3528 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3552 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3552 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3576 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3576 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3600 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3600 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3624 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1176 + i ], ((simde__m128i*) bnProcBuf)[3624 + i]);
}
// Process group with 17 CNs 
// Process group with 18 CNs 
// Process group with 19 CNs 
// Process group with 20 CNs 
// Process group with 21 CNs 
// Process group with 22 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3648 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3648 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3672 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3672 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3696 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3696 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3720 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3720 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3744 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3744 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3768 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3768 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3792 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3792 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3816 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3816 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3840 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3840 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3864 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3864 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3888 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3888 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3912 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3912 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3936 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3936 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3960 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3960 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3984 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[3984 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4008 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[4008 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4032 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[4032 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4056 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[4056 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4080 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[4080 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4104 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[4104 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4128 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[4128 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4152 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[4152 + i]);
}
// Process group with <23 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4176 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4176 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4200 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4200 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4224 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4224 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4248 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4248 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4272 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4272 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4296 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4296 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4320 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4320 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4344 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4344 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4368 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4368 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4392 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4392 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4416 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4416 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4440 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4440 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4464 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4464 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4488 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4488 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4512 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4512 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4536 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4536 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4560 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4560 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4584 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4584 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4608 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4608 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4632 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4632 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4656 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4656 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4680 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4680 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4704 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1224 + i ], ((simde__m128i*) bnProcBuf)[4704 + i]);
}
// Process group with 24 CNs 
// Process group with 25 CNs 
// Process group with 26 CNs 
// Process group with 27 CNs 
// Process group with 28 CNs 
// Process group with 29 CNs 
// Process group with 30 CNs 
}
