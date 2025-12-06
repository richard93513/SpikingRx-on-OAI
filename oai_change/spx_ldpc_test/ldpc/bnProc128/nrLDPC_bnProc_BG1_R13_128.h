static inline void nrLDPC_bnProc_BG1_R13_128(int8_t* bnProcBuf,int8_t* bnProcBufRes,  int8_t* llrRes, uint16_t Z ) {
        uint32_t M, i; 
// Process group with 2 CNs 
// Process group with 3 CNs 
// Process group with 4 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1008 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1008 + i ], ((simde__m128i*) bnProcBuf)[1008 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1032 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1008 + i ], ((simde__m128i*) bnProcBuf)[1032 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1056 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1008 + i ], ((simde__m128i*) bnProcBuf)[1056 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1080 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1008 + i ], ((simde__m128i*) bnProcBuf)[1080 + i]);
}
// Process group with 5 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1104 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[1104 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1128 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[1128 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1152 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[1152 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1176 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[1176 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1200 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1032 + i ], ((simde__m128i*) bnProcBuf)[1200 + i]);
}
// Process group with 6 CNs 
       M = (2*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1224 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1056 + i ], ((simde__m128i*) bnProcBuf)[1224 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1272 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1056 + i ], ((simde__m128i*) bnProcBuf)[1272 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1320 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1056 + i ], ((simde__m128i*) bnProcBuf)[1320 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1368 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1056 + i ], ((simde__m128i*) bnProcBuf)[1368 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1416 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1056 + i ], ((simde__m128i*) bnProcBuf)[1416 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1464 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1056 + i ], ((simde__m128i*) bnProcBuf)[1464 + i]);
}
// Process group with 7 CNs 
       M = (4*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1512 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[1512 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1608 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[1608 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1704 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[1704 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1800 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[1800 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1896 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[1896 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1992 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[1992 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2088 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1104 + i ], ((simde__m128i*) bnProcBuf)[2088 + i]);
}
// Process group with 8 CNs 
       M = (3*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2184 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[2184 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2256 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[2256 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2328 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[2328 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2400 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[2400 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2472 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[2472 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2544 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[2544 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2616 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[2616 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2688 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1200 + i ], ((simde__m128i*) bnProcBuf)[2688 + i]);
}
// Process group with 9 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2760 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1272 + i ], ((simde__m128i*) bnProcBuf)[2760 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2784 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1272 + i ], ((simde__m128i*) bnProcBuf)[2784 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2808 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1272 + i ], ((simde__m128i*) bnProcBuf)[2808 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2832 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1272 + i ], ((simde__m128i*) bnProcBuf)[2832 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2856 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1272 + i ], ((simde__m128i*) bnProcBuf)[2856 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2880 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1272 + i ], ((simde__m128i*) bnProcBuf)[2880 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2904 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1272 + i ], ((simde__m128i*) bnProcBuf)[2904 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2928 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1272 + i ], ((simde__m128i*) bnProcBuf)[2928 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2952 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1272 + i ], ((simde__m128i*) bnProcBuf)[2952 + i]);
}
// Process group with 10 CNs 
       M = (4*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2976 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1296 + i ], ((simde__m128i*) bnProcBuf)[2976 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3072 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1296 + i ], ((simde__m128i*) bnProcBuf)[3072 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3168 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1296 + i ], ((simde__m128i*) bnProcBuf)[3168 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3264 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1296 + i ], ((simde__m128i*) bnProcBuf)[3264 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3360 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1296 + i ], ((simde__m128i*) bnProcBuf)[3360 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3456 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1296 + i ], ((simde__m128i*) bnProcBuf)[3456 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3552 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1296 + i ], ((simde__m128i*) bnProcBuf)[3552 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3648 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1296 + i ], ((simde__m128i*) bnProcBuf)[3648 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3744 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1296 + i ], ((simde__m128i*) bnProcBuf)[3744 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3840 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1296 + i ], ((simde__m128i*) bnProcBuf)[3840 + i]);
}
// Process group with 11 CNs 
       M = (3*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[3936 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1392 + i ], ((simde__m128i*) bnProcBuf)[3936 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4008 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1392 + i ], ((simde__m128i*) bnProcBuf)[4008 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4080 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1392 + i ], ((simde__m128i*) bnProcBuf)[4080 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4152 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1392 + i ], ((simde__m128i*) bnProcBuf)[4152 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4224 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1392 + i ], ((simde__m128i*) bnProcBuf)[4224 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4296 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1392 + i ], ((simde__m128i*) bnProcBuf)[4296 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4368 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1392 + i ], ((simde__m128i*) bnProcBuf)[4368 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4440 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1392 + i ], ((simde__m128i*) bnProcBuf)[4440 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4512 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1392 + i ], ((simde__m128i*) bnProcBuf)[4512 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4584 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1392 + i ], ((simde__m128i*) bnProcBuf)[4584 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4656 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1392 + i ], ((simde__m128i*) bnProcBuf)[4656 + i]);
}
// Process group with 12 CNs 
       M = (4*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4728 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1464 + i ], ((simde__m128i*) bnProcBuf)[4728 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4824 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1464 + i ], ((simde__m128i*) bnProcBuf)[4824 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[4920 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1464 + i ], ((simde__m128i*) bnProcBuf)[4920 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5016 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1464 + i ], ((simde__m128i*) bnProcBuf)[5016 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5112 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1464 + i ], ((simde__m128i*) bnProcBuf)[5112 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5208 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1464 + i ], ((simde__m128i*) bnProcBuf)[5208 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5304 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1464 + i ], ((simde__m128i*) bnProcBuf)[5304 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5400 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1464 + i ], ((simde__m128i*) bnProcBuf)[5400 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5496 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1464 + i ], ((simde__m128i*) bnProcBuf)[5496 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5592 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1464 + i ], ((simde__m128i*) bnProcBuf)[5592 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5688 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1464 + i ], ((simde__m128i*) bnProcBuf)[5688 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5784 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1464 + i ], ((simde__m128i*) bnProcBuf)[5784 + i]);
}
// Process group with 13 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5880 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1560 + i ], ((simde__m128i*) bnProcBuf)[5880 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5904 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1560 + i ], ((simde__m128i*) bnProcBuf)[5904 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5928 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1560 + i ], ((simde__m128i*) bnProcBuf)[5928 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5952 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1560 + i ], ((simde__m128i*) bnProcBuf)[5952 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[5976 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1560 + i ], ((simde__m128i*) bnProcBuf)[5976 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6000 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1560 + i ], ((simde__m128i*) bnProcBuf)[6000 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6024 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1560 + i ], ((simde__m128i*) bnProcBuf)[6024 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6048 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1560 + i ], ((simde__m128i*) bnProcBuf)[6048 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6072 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1560 + i ], ((simde__m128i*) bnProcBuf)[6072 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6096 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1560 + i ], ((simde__m128i*) bnProcBuf)[6096 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6120 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1560 + i ], ((simde__m128i*) bnProcBuf)[6120 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6144 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1560 + i ], ((simde__m128i*) bnProcBuf)[6144 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6168 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1560 + i ], ((simde__m128i*) bnProcBuf)[6168 + i]);
}
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
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6192 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6192 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6216 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6216 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6240 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6240 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6264 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6264 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6288 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6288 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6312 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6312 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6336 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6336 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6360 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6360 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6384 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6384 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6408 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6408 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6432 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6432 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6456 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6456 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6480 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6480 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6504 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6504 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6528 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6528 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6552 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6552 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6576 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6576 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6600 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6600 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6624 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6624 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6648 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6648 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6672 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6672 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6696 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6696 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6720 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6720 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6744 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6744 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6768 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6768 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6792 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6792 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6816 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6816 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6840 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1584 + i ], ((simde__m128i*) bnProcBuf)[6840 + i]);
}
// Process group with 29 CNs 
// Process group with 30 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6864 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[6864 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6888 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[6888 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6912 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[6912 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6936 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[6936 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6960 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[6960 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[6984 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[6984 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7008 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7008 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7032 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7032 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7056 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7056 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7080 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7080 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7104 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7104 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7128 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7128 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7152 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7152 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7176 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7176 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7200 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7200 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7224 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7224 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7248 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7248 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7272 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7272 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7296 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7296 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7320 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7320 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7344 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7344 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7368 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7368 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7392 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7392 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7416 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7416 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7440 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7440 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7464 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7464 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7488 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7488 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7512 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7512 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7536 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7536 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[7560 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[1608 + i ], ((simde__m128i*) bnProcBuf)[7560 + i]);
}
}
