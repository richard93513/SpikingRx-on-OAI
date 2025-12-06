static inline void nrLDPC_bnProc_BG1_R89_128(int8_t* bnProcBuf,int8_t* bnProcBufRes,  int8_t* llrRes, uint16_t Z ) {
        uint32_t M, i; 
// Process group with 2 CNs 
 M = (3*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[24 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[24 + i ], ((simde__m128i*) bnProcBuf)[24 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[96 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[24 + i ], ((simde__m128i*) bnProcBuf)[96 + i]);
}
// Process group with 3 CNs 
       M = (21*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[168 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[96 + i ], ((simde__m128i*) bnProcBuf)[168 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[672 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[96 + i ], ((simde__m128i*) bnProcBuf)[672 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1176 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[96 + i ], ((simde__m128i*) bnProcBuf)[1176 + i]);
}
// Process group with 4 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1680 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[600 + i ], ((simde__m128i*) bnProcBuf)[1680 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1704 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[600 + i ], ((simde__m128i*) bnProcBuf)[1704 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1728 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[600 + i ], ((simde__m128i*) bnProcBuf)[1728 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1752 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[600 + i ], ((simde__m128i*) bnProcBuf)[1752 + i]);
}
// Process group with 5 CNs 
       M = (1*Z + 15)>>4;
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1776 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[624 + i ], ((simde__m128i*) bnProcBuf)[1776 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1800 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[624 + i ], ((simde__m128i*) bnProcBuf)[1800 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1824 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[624 + i ], ((simde__m128i*) bnProcBuf)[1824 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1848 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[624 + i ], ((simde__m128i*) bnProcBuf)[1848 + i]);
}
            for (i=0;i<M;i++) {
            ((simde__m128i*)bnProcBufRes)[1872 + i ] = simde_mm_subs_epi8(((simde__m128i*)llrRes)[624 + i ], ((simde__m128i*) bnProcBuf)[1872 + i]);
}
// Process group with 6 CNs 
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
