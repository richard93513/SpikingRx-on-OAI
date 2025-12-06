// spx_ldpc_test.c
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "ldpc/nrLDPC_types.h"
#include "ldpc/nrLDPCdecoder_defs.h"
#include "ldpc/nrLDPC_init.h"
#include "ldpc/nrLDPC_lut.h"

int32_t nrLDPC_decoder(t_nrLDPC_dec_params *p_decParams,
                       int8_t              *p_llr,
                       int8_t              *p_out);

// 簡單 JSON 取整數的小工具：在文字裡找 "key": 123
static int json_get_int(const char *buf, const char *key, int *out)
{
    const char *p = strstr(buf, key);
    if (!p)
        return -1;

    p = strchr(p, ':');
    if (!p)
        return -1;

    p++; // 跳過 ':'
    while (*p == ' ' || *p == '\t')
        p++;

    int sign = 1;
    if (*p == '-') {
        sign = -1;
        p++;
    }

    if (*p < '0' || *p > '9')
        return -1;

    long val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }

    *out = (int)(sign * val);
    return 0;
}

typedef struct {
    int BG;
    int Zc;
    int A;
    int G;
    int C;
    int Qm;
    int F;
    int rv_index;
    int tbslbrm;
    int E[16];  // 先放 E0, E1 ...
} ldpc_cfg_t;

// 讀 ldpc_cfg.json
static int load_ldpc_cfg(const char *path, ldpc_cfg_t *cfg)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] open cfg '%s' fail: %s\n", path, strerror(errno));
        return -1;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "[ERROR] fseek(cfg) fail\n");
        fclose(f);
        return -1;
    }
    long len = ftell(f);
    if (len <= 0) {
        fprintf(stderr, "[ERROR] cfg file too small\n");
        fclose(f);
        return -1;
    }
    rewind(f);

    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) {
        fprintf(stderr, "[ERROR] malloc(%ld) fail\n", len);
        fclose(f);
        return -1;
    }

    size_t r = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[r] = '\0';

    // 初始化
    memset(cfg, 0, sizeof(*cfg));

    // 依照你現在的 ldpc_cfg.json 欄位名抓數字
    if (json_get_int(buf, "\"BG\"", &cfg->BG) != 0) {
        fprintf(stderr, "[ERROR] parse BG from cfg fail\n");
        free(buf);
        return -1;
    }
    json_get_int(buf, "\"Zc\"", &cfg->Zc);
    json_get_int(buf, "\"A\"", &cfg->A);
    json_get_int(buf, "\"G\"", &cfg->G);
    json_get_int(buf, "\"C\"", &cfg->C);
    json_get_int(buf, "\"Qm\"", &cfg->Qm);
    json_get_int(buf, "\"F\"", &cfg->F);
    json_get_int(buf, "\"rv_index\"", &cfg->rv_index);
    json_get_int(buf, "\"tbslbrm\"", &cfg->tbslbrm);

    // E0, E1 (如果有的話就抓起來；沒有就保持 0)
    json_get_int(buf, "\"E0\"", &cfg->E[0]);
    json_get_int(buf, "\"E1\"", &cfg->E[1]);

    free(buf);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr,
                "Usage: %s <infer_llr_int8.bin> <ldpc_cfg.json> <decoded_bits.bin>\n",
                argv[0]);
        return 1;
    }

    const char *llr_path = argv[1];
    const char *cfg_path = argv[2];
    const char *out_path = argv[3];

    printf("SPX LDPC TEST: standalone build OK!\n");

    // -------------------------------------------------
    // 1) 讀 cfg
    // -------------------------------------------------
    ldpc_cfg_t cfg;
    if (load_ldpc_cfg(cfg_path, &cfg) != 0) {
        fprintf(stderr, "[ERROR] load_ldpc_cfg failed\n");
        return 1;
    }

    printf("[CFG] BG=%d, Zc=%d, A=%d, G=%d, C=%d, Qm=%d\n",
           cfg.BG, cfg.Zc, cfg.A, cfg.G, cfg.C, cfg.Qm);
    printf("[CFG] F=%d, rv_index=%d, tbslbrm=%d\n",
           cfg.F, cfg.rv_index, cfg.tbslbrm);
    printf("[CFG] E0=%d, E1=%d\n", cfg.E[0], cfg.E[1]);

    // -------------------------------------------------
    // 2) 檢查 LLR 檔長度
    // -------------------------------------------------
    FILE *f_llr = fopen(llr_path, "rb");
    if (!f_llr) {
        fprintf(stderr, "[ERROR] open llr '%s' fail: %s\n", llr_path, strerror(errno));
        return 1;
    }

    if (fseek(f_llr, 0, SEEK_END) != 0) {
        fprintf(stderr, "[ERROR] fseek(llr) fail\n");
        fclose(f_llr);
        return 1;
    }
    long llr_len = ftell(f_llr);
    if (llr_len < 0) {
        fprintf(stderr, "[ERROR] ftell(llr) fail\n");
        fclose(f_llr);
        return 1;
    }

    printf("[LLR] file bytes = %ld\n", llr_len);

    // 預期 LLR 數量：E0 如果有就用 E0，否則退而求其次用 G
    int E = (cfg.E[0] > 0) ? cfg.E[0] : cfg.G;
    if (E <= 0) {
        fprintf(stderr, "[WARN] E0/G 都沒有合理數值 (E=%d)，先照檔案長度走\n", E);
    } else {
        printf("[LLR] expect %d LLRs (E)\n", E);
        if (llr_len != E) {
            fprintf(stderr,
                    "[WARN] LLR 檔案長度 (%ld) 和 E (%d) 不一致，"
                    "解碼可能會失敗或需要檢查 pipeline\n",
                    llr_len, E);
        }
    }

    // 重新回到檔案開頭，讀進記憶體
    rewind(f_llr);

    int8_t *llr_buf = (int8_t *)malloc((size_t)llr_len);
    if (!llr_buf) {
        fprintf(stderr, "[ERROR] malloc(%ld) for LLR fail\n", llr_len);
        fclose(f_llr);
        return 1;
    }

    size_t nread = fread(llr_buf, 1, (size_t)llr_len, f_llr);
    fclose(f_llr);

    if (nread != (size_t)llr_len) {
        fprintf(stderr, "[ERROR] fread(llr) only %zu/%ld bytes\n", nread, llr_len);
        free(llr_buf);
        return 1;
    }

    printf("[LLR] loaded %ld bytes of LLR from '%s'\n", llr_len, llr_path);

    // -------------------------------------------------
    // 3) 建立輸出 buffer（暫時先全部填 0）
    //    之後在這裡接 OAI 的 nrLDPC_decoder()
    // -------------------------------------------------
    int A = cfg.A;         // TB bits（不含 CRC）
    int tb_bits  = (A > 0) ? A : 0;

    if (tb_bits <= 0) {
        fprintf(stderr,
                "[WARN] cfg.A = %d，看起來沒設定 TB bits，"
                "先寫出 0 byte 的 decoded 檔（只是測 I/O）\n",
                cfg.A);
    }

    // 注意：因為 outMode = nrLDPC_outMode_BITINT8
    // → 1 bit 對應 1 個 int8 output
    // 所以這裡直接開 tb_bits 個 byte
    int decoded_len_bytes = tb_bits;

    uint8_t *decoded_bits = NULL;
    if (decoded_len_bytes > 0) {
        decoded_bits = (uint8_t *)calloc((size_t)decoded_len_bytes, 1);
        if (!decoded_bits) {
            fprintf(stderr, "[ERROR] malloc decoded_bits(%d bytes) fail\n", decoded_len_bytes);
            free(llr_buf);
            return 1;
        }
    }


    // -------------------------------------------------
    // 3) 呼叫 OAI LDPC decoder：llr_buf → decoded_bits
    // -------------------------------------------------
    if (tb_bits > 0) {
        t_nrLDPC_dec_params decParams;
        memset(&decParams, 0, sizeof(decParams));

        // 基本圖與 lifting size
        decParams.BG         = (uint8_t)cfg.BG;
        decParams.Z          = (uint16_t)cfg.Zc;

        // 最大迭代數
        decParams.numMaxIter = 25;

        // 輸出格式：每 bit 一個 int8
        decParams.outMode    = nrLDPC_outMode_BITINT8;

        // ---------- 計算母碼長度 N ----------
        int N_ldpc;
        if (cfg.BG == 1) {
            // 5G NR, BG1: N = 66 * Zc
            N_ldpc = 66 * cfg.Zc;
        } else {
            // 5G NR, BG2: N = 50 * Zc
            N_ldpc = 50 * cfg.Zc;
        }

        printf("[DEC] BG=%d, Zc=%d → LDPC mother code length N=%d\n",
               cfg.BG, cfg.Zc, N_ldpc);

        // 檢查一下 G vs N
        if (cfg.G > N_ldpc) {
            fprintf(stderr,
                    "[WARN] G=%d > N_ldpc=%d，rate-matching 未處理，"
                    "decoder 可能仍然表現很差（但不該 segfault）。\n",
                    cfg.G, N_ldpc);
        }

        // ---------- 建新的 LLR buffer，長度 N_ldpc ----------
        int8_t *llr_ldpc = (int8_t *)calloc((size_t)N_ldpc, 1);
        if (!llr_ldpc) {
            fprintf(stderr, "[ERROR] malloc llr_ldpc(%d) fail\n", N_ldpc);
            free(llr_buf);
            free(decoded_bits);
            return 1;
        }

        // 把你真實的 LLR 拷到前面，剩下的補 0
        int copy_len = (cfg.G < N_ldpc) ? cfg.G : N_ldpc;
        memcpy(llr_ldpc, llr_buf, (size_t)copy_len);

        // 初始化 decoder（建 lookup table 等）
        nrLDPC_init(&decParams, NULL);

        // === 真正呼叫 LDPC decoder，丟母碼長度 N 的 LLR ===
        int32_t ldpc_ret = nrLDPC_decoder(&decParams,
                                          llr_ldpc,
                                          (int8_t *)decoded_bits);

        free(llr_ldpc);  // 用完釋放

        if (ldpc_ret < 0) {
            fprintf(stderr, "[ERROR] nrLDPC_decoder() return %d\n", ldpc_ret);
            free(llr_buf);
            free(decoded_bits);
            return 1;
        }

        printf("[DEC] nrLDPC_decoder() done, ret = %d\n", ldpc_ret);
    }



    FILE *f_out = fopen(out_path, "wb");
    if (!f_out) {
        fprintf(stderr, "[ERROR] open out '%s' fail: %s\n", out_path, strerror(errno));
        free(llr_buf);
        free(decoded_bits);
        return 1;
    }

    size_t nw = 0;
    if (decoded_len_bytes > 0) {
        nw = fwrite(decoded_bits, 1, (size_t)decoded_len_bytes, f_out);
    }
    fclose(f_out);

    if (decoded_len_bytes > 0 && nw != (size_t)decoded_len_bytes) {
        fprintf(stderr, "[ERROR] fwrite(decoded_bits) only %zu/%d bytes\n",
                nw, decoded_len_bytes);
        free(llr_buf);
        free(decoded_bits);
        return 1;
    }

    printf("[OK] wrote %d bits as %d int8-bytes to '%s'\n",
           tb_bits, decoded_len_bytes, out_path);


    free(llr_buf);
    free(decoded_bits);

    return 0;
}

