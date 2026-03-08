// -----------------------------------------------------------------
// spx_ldpc_test.c
// 最小 OAI LDPC decoder 測試：
// - 填 decParams
// - 呼叫 nrLDPC_init()
// - 呼叫 nrLDPC_decod()
// -----------------------------------------------------------------

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "PHY/CODING/nrLDPC_decoder/nrLDPC_types.h"
#include "PHY/CODING/nrLDPC_decoder/nrLDPC_init.h"
#include "PHY/CODING/nrLDPC_extern.h"

// decoder 函式（OAI 提供）
extern int32_t nrLDPC_decod(t_nrLDPC_dec_params *p_decParams,
                             int8_t *p_llr,
                             int8_t *p_out);

int main() {

    printf("=== SPX LDPC DECODER TEST ===\n");

    // ------------------------------------------------------------
    // Step 1: 準備 decParams
    // ------------------------------------------------------------
    t_nrLDPC_dec_params decParams;
    memset(&decParams, 0, sizeof(decParams));

    // 先用一組固定的合法設定 (BG=1, Zc=384)
    decParams.BG         = 1;      // Base graph 1
    decParams.Z          = 384;    // Lifting size
    decParams.Kprime     = 8448;   // TB segment length (info bits)
    decParams.numMaxIter = 10;     // decoder 最大迭代
    decParams.outMode    = nrLDPC_outMode_BIT;  // 輸出 hard bits
    decParams.R          = 0;      // 不重要，後面根據 JSON 再補

    printf("decParams: BG=%d Z=%d Kprime=%d\n",
           decParams.BG, decParams.Z, decParams.Kprime);

    // ------------------------------------------------------------
    // Step 2: 呼叫 nrLDPC_init → 得到需要的 LLR 數量
    // ------------------------------------------------------------
    uint32_t llr_len = nrLDPC_init(&decParams);

    if (llr_len == 0) {
        printf("ERROR: nrLDPC_init returned 0 → decParams 可能無效！\n");
        return -1;
    }

    printf("nrLDPC_init → llr_len = %u\n", llr_len);

    // ------------------------------------------------------------
    // Step 3: 準備假的 LLR（全 +4）當作 input
    // ------------------------------------------------------------
    int8_t *llr  = malloc(llr_len);
    int8_t *bits = malloc(llr_len);

    for (uint32_t i = 0; i < llr_len; i++)
        llr[i] = 4;      // 全 0 bits，高信心

    memset(bits, 0, llr_len);

    // ------------------------------------------------------------
    // Step 4: 呼叫 decoder
    // ------------------------------------------------------------
    int32_t iters = nrLDPC_decod(&decParams, llr, bits);

    printf("Decoder used iterations = %d\n", iters);

    // ------------------------------------------------------------
    // Step 5: 印前 32 個 bits
    // ------------------------------------------------------------
    printf("First 32 decoded bits:\n");
    for (int i = 0; i < 32; i++) {
        printf("%d", bits[i] & 1);
    }
    printf("\n");

    free(llr);
    free(bits);

    return 0;
}


