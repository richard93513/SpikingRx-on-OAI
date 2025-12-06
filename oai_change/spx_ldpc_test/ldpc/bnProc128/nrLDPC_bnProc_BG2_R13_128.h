static inline void nrLDPC_bnProc_BG2_R13_128(int8_t* bnProcBuf,int8_t* bnProcBufRes,  int8_t* llrRes, uint16_t Z  ) {
        uint32_t M, i; 
// Process group with 2 CNs 
 M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[432 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[432 + i ], ((simde__m128i*) bnProcBuf)[432 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[456 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[432 + i ], ((simde__m128i*) bnProcBuf)[456 + i]);
}
// Process group with 3 CNs 
// Process group with 4 CNs 
       M = (2*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[480 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[456 + i ], ((simde__m128i*) bnProcBuf)[480 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[528 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[456 + i ], ((simde__m128i*) bnProcBuf)[528 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[576 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[456 + i ], ((simde__m128i*) bnProcBuf)[576 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[624 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[456 + i ], ((simde__m128i*) bnProcBuf)[624 + i]);
}
// Process group with 5 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[672 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[504 + i ], ((simde__m128i*) bnProcBuf)[672 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[696 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[504 + i ], ((simde__m128i*) bnProcBuf)[696 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[720 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[504 + i ], ((simde__m128i*) bnProcBuf)[720 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[744 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[504 + i ], ((simde__m128i*) bnProcBuf)[744 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[768 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[504 + i ], ((simde__m128i*) bnProcBuf)[768 + i]);
}
// Process group with 6 CNs 
       M = (5*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[792 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[528 + i ], ((simde__m128i*) bnProcBuf)[792 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[912 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[528 + i ], ((simde__m128i*) bnProcBuf)[912 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1032 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[528 + i ], ((simde__m128i*) bnProcBuf)[1032 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1152 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[528 + i ], ((simde__m128i*) bnProcBuf)[1152 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1272 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[528 + i ], ((simde__m128i*) bnProcBuf)[1272 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1392 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[528 + i ], ((simde__m128i*) bnProcBuf)[1392 + i]);
}
// Process group with 7 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1512 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[648 + i ], ((simde__m128i*) bnProcBuf)[1512 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1536 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[648 + i ], ((simde__m128i*) bnProcBuf)[1536 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1560 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[648 + i ], ((simde__m128i*) bnProcBuf)[1560 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1584 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[648 + i ], ((simde__m128i*) bnProcBuf)[1584 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1608 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[648 + i ], ((simde__m128i*) bnProcBuf)[1608 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1632 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[648 + i ], ((simde__m128i*) bnProcBuf)[1632 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1656 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[648 + i ], ((simde__m128i*) bnProcBuf)[1656 + i]);
}
// Process group with 8 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1680 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[672 + i ], ((simde__m128i*) bnProcBuf)[1680 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1704 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[672 + i ], ((simde__m128i*) bnProcBuf)[1704 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1728 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[672 + i ], ((simde__m128i*) bnProcBuf)[1728 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1752 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[672 + i ], ((simde__m128i*) bnProcBuf)[1752 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1776 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[672 + i ], ((simde__m128i*) bnProcBuf)[1776 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1800 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[672 + i ], ((simde__m128i*) bnProcBuf)[1800 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1824 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[672 + i ], ((simde__m128i*) bnProcBuf)[1824 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1848 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[672 + i ], ((simde__m128i*) bnProcBuf)[1848 + i]);
}
// Process group with 9 CNs 
// Process group with 10 CNs 
// Process group with 11 CNs 
// Process group with 12 CNs 
// Process group with 13 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1872 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[696 + i ], ((simde__m128i*) bnProcBuf)[1872 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1896 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[696 + i ], ((simde__m128i*) bnProcBuf)[1896 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1920 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[696 + i ], ((simde__m128i*) bnProcBuf)[1920 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1944 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[696 + i ], ((simde__m128i*) bnProcBuf)[1944 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1968 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[696 + i ], ((simde__m128i*) bnProcBuf)[1968 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1992 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[696 + i ], ((simde__m128i*) bnProcBuf)[1992 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2016 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[696 + i ], ((simde__m128i*) bnProcBuf)[2016 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2040 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[696 + i ], ((simde__m128i*) bnProcBuf)[2040 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2064 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[696 + i ], ((simde__m128i*) bnProcBuf)[2064 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2088 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[696 + i ], ((simde__m128i*) bnProcBuf)[2088 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2112 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[696 + i ], ((simde__m128i*) bnProcBuf)[2112 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2136 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[696 + i ], ((simde__m128i*) bnProcBuf)[2136 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2160 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[696 + i ], ((simde__m128i*) bnProcBuf)[2160 + i]);
}
// Process group with 14 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2184 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2184 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2208 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2208 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2232 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2232 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2256 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2256 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2280 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2280 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2304 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2304 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2328 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2328 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2352 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2352 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2376 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2376 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2400 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2400 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2424 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2424 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2448 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2448 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2472 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2472 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2496 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[720 + i ], ((simde__m128i*) bnProcBuf)[2496 + i]);
}
// Process group with 15 CNs 
// Process group with 16 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2520 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2520 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2544 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2544 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2568 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2568 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2592 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2592 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2616 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2616 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2640 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2640 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2664 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2664 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2688 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2688 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2712 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2712 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2736 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2736 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2760 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2760 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2784 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2784 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2808 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2808 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2832 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2832 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2856 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2856 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[2880 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[744 + i ], ((simde__m128i*) bnProcBuf)[2880 + i]);
}
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
