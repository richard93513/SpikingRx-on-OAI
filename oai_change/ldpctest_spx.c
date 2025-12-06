/*
 * ldpctest_spx.c
 *
 * 專門給 SpikingRx 用的 LDPC decoder 小工具：
 *
 * 用法：
 *
 *   ldpctest_spx <infer_llr_int8.bin> <ldpc_cfg.txt> <decoded_bits.bin>
 *
 * 其中 <ldpc_cfg.txt> 格式為一行一個欄位，例如：
 *
 *   frame   278
 *   slot    11
 *   dlsch_id        0
 *   BG      1
 *   Zc      224
 *   A       9480
 *   C       2
 *   K       4928
 *   F       152
 *   G       14400
 *   Qm      2
 *   nb_layers       1
 *   rv_index        3
 *   tbslbrm 184424
 *   mcs     9
 *
 * 假設：
 *   - LLR 檔長度必為 G（你說「LLR always length == G」）。
 *   - G 是單一 TB、單 codeword 的總 LLR 數。
 *   - C 是 codeblock 數（segmentation）。
 *
 * 這個工具：
 *   1. 讀 cfg → 取出 BG, Zc, A, C, F, G, Qm, rv_index...
 *   2. 推出每個 codeblock 的 Kprime（含 CB-CRC、含 filler，不含 parity）。
 *   3. 假設每個 codeblock 的 rate-matched 輸入長度 E = G / C。
 *   4. 估計實際 code rate，對應到 OAI 的 decParams.R。
 *   5. 將 LLR 均分成 C 段，每段長度 E，逐段呼叫 OAI 的 LDPCdecoder。
 *   6. LDPCdecoder 輸出的 bit 仍為 bit-packed（1 byte 8 bit）→ 這裡展開成「1 bit → 1 byte 的 0/1」，
 *      依序寫到 decoded_bits.bin。
 *
 * 注意：
 *   - 輸出長度為 C * Kprime bytes（每個 codeblock Kprime bit 展開、一個 bit 一 byte）。
 *   - 還沒做「還原 TB 的 A bit」（也就是沒把所有 CB-CRC 去掉、filler 去掉再重組成 TB）。
 *     要做到這一步，建議你在這個骨架上對照 openair1/PHY/NR_UE_TRANSPORT/nr_dlsch_decoding.c 來補。
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <stdbool.h>   // <<< 新增：為了 bool
#include <malloc.h>    // <<< 新增：為了 memalign / malloc16

#include "assertions.h"
#include "SIMULATION/TOOLS/sim.h"
#include "common/config/config_userapi.h"
#include "common/utils/load_module_shlib.h"
#include "common/utils/LOG/log.h"
#include "openair1/PHY/defs_nr_common.h"
#include "PHY/CODING/nrLDPC_extern.h"

#include "coding_unitary_defs.h"

#ifndef malloc16
#define malloc16(x) memalign(32, x)
#endif

#define MAX_BLOCK_LENGTH 8448
#define MAX_SEGMENTS     MAX_NUM_DLSCH_SEGMENTS
#define MAX_LLR_LEN      (68 * 384)  /* 跟原 ldpctest.c 相同的最大長度 */

static ldpc_interface_t ldpc_toCompare;

/* 簡單 key-value 文字檔 parser：
 * 讀一行，例如 "BG\t1" 或 "BG 1" 或 "BG    1"
 * 用 sscanf("%127s %d", key, &val) 解析。
 */
typedef struct {
  int frame;
  int slot;
  int dlsch_id;
  int BG;
  int Zc;
  int A;
  int C;
  int K;
  int F;
  int G;
  int Qm;
  int nb_layers;
  int rv_index;
  int tbslbrm;
  int mcs;
} ldpc_cfg_t;

static int load_ldpc_cfg_kv(const char *path, ldpc_cfg_t *cfg)
{
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "[CFG] open '%s' fail: %s\n", path, strerror(errno));
    return -1;
  }

  memset(cfg, 0, sizeof(*cfg));

  char line[256];
  char key[128];
  int val;
  while (fgets(line, sizeof(line), f)) {
    if (sscanf(line, "%127s %d", key, &val) != 2)
      continue;

    if      (strcmp(key, "frame")      == 0) cfg->frame      = val;
    else if (strcmp(key, "slot")       == 0) cfg->slot       = val;
    else if (strcmp(key, "dlsch_id")   == 0) cfg->dlsch_id   = val;
    else if (strcmp(key, "BG")         == 0) cfg->BG         = val;
    else if (strcmp(key, "Zc")         == 0) cfg->Zc         = val;
    else if (strcmp(key, "A")          == 0) cfg->A          = val;
    else if (strcmp(key, "C")          == 0) cfg->C          = val;
    else if (strcmp(key, "K")          == 0) cfg->K          = val;
    else if (strcmp(key, "F")          == 0) cfg->F          = val;
    else if (strcmp(key, "G")          == 0) cfg->G          = val;
    else if (strcmp(key, "Qm")         == 0) cfg->Qm         = val;
    else if (strcmp(key, "nb_layers")  == 0) cfg->nb_layers  = val;
    else if (strcmp(key, "rv_index")   == 0) cfg->rv_index   = val;
    else if (strcmp(key, "tbslbrm")    == 0) cfg->tbslbrm    = val;
    else if (strcmp(key, "mcs")        == 0) cfg->mcs        = val;
  }

  fclose(f);

  if (cfg->BG == 0 || cfg->Zc == 0 || cfg->A == 0 || cfg->C == 0 || cfg->G == 0) {
    fprintf(stderr,
            "[CFG] invalid cfg: BG=%d Zc=%d A=%d C=%d G=%d (至少這幾個要有)\n",
            cfg->BG, cfg->Zc, cfg->A, cfg->C, cfg->G);
    return -1;
  }

  printf("[CFG] frame=%d slot=%d dlsch_id=%d\n", cfg->frame, cfg->slot, cfg->dlsch_id);
  printf("[CFG] BG=%d Zc=%d A=%d C=%d K=%d F=%d G=%d Qm=%d nb_layers=%d\n",
         cfg->BG, cfg->Zc, cfg->A, cfg->C, cfg->K, cfg->F, cfg->G,
         cfg->Qm, cfg->nb_layers);
  printf("[CFG] rv_index=%d tbslbrm=%d mcs=%d\n",
         cfg->rv_index, cfg->tbslbrm, cfg->mcs);

  return 0;
}

/* 讀 LLR 檔，檢查長度 == cfg->G */
static int8_t *load_llr_file(const char *path, const ldpc_cfg_t *cfg, int *out_len)
{
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "[LLR] open '%s' fail: %s\n", path, strerror(errno));
    return NULL;
  }

  if (fseek(f, 0, SEEK_END) != 0) {
    fprintf(stderr, "[LLR] fseek fail\n");
    fclose(f);
    return NULL;
  }
  long len = ftell(f);
  if (len <= 0) {
    fprintf(stderr, "[LLR] file too small (%ld)\n", len);
    fclose(f);
    return NULL;
  }
  rewind(f);

  printf("[LLR] file bytes = %ld (expect G=%d)\n", len, cfg->G);
  if (len != cfg->G) {
    fprintf(stderr,
            "[LLR] WARNING: file length (%ld) != G (%d)，會以檔案長度為準\n",
            len, cfg->G);
  }

  int8_t *buf = (int8_t *)malloc(len);
  if (!buf) {
    fprintf(stderr, "[LLR] malloc(%ld) fail\n", len);
    fclose(f);
    return NULL;
  }

  size_t n = fread(buf, 1, len, f);
  fclose(f);

  if (n != (size_t)len) {
    fprintf(stderr, "[LLR] fread only %zu/%ld bytes\n", n, len);
    free(buf);
    return NULL;
  }

  *out_len = (int)len;
  return buf;
}

/* 根據 A, C, F 推 Kprime（每個 codeblock 的 bits，含 CB-CRC & filler、不含 parity）
 * 3GPP 38.212 segmentation (簡化)：
 *   B      = A + 24            // TB 加 TB-CRC
 *   B'     = B + 24*C         // 每個 CB 再加 24-bit CRC
 *   C*K'   = B' + F           // F filler bits
 * → K' = (B' + F) / C
 */
static int derive_Kprime(const ldpc_cfg_t *cfg)
{
  int A = cfg->A;
  int C = cfg->C;
  int F = cfg->F;

  int B  = A + 24;
  int Bp = B + 24 * C;
  int numerator = Bp + F;

  if (numerator % C != 0) {
    fprintf(stderr,
            "[DERIVE] (B' + F) = %d 不能整除 C=%d，Kprime 非整數，這組參數怪怪的\n",
            numerator, C);
    return -1;
  }

  int Kprime = numerator / C;
  printf("[DERIVE] B=%d B'=%d F=%d → Kprime=%d (每個 codeblock 的 bits, 含 CB-CRC)\n",
         B, Bp, F, Kprime);

  /* 檢查 Kprime 是否符合 NR 限制 */
  int Kcb = (cfg->BG == 1) ? 8448 : 3840;
  if (Kprime > Kcb) {
    fprintf(stderr,
            "[DERIVE] Kprime=%d > Kcb=%d (BG=%d)，不符合規範\n",
            Kprime, Kcb, cfg->BG);
    /* 先警告，但仍回傳，讓你自己決定要不要硬 decode */
  }

  return Kprime;
}

/* 估計實際 code rate，選出 OAI 的 code_rate_vec index
 * 本函式會回傳 decParams.R 要填的值（code_rate_vec[R_ind]），以及 R_ind 本身。
 */
static int pick_ldpc_R(const ldpc_cfg_t *cfg, int Kprime, int *out_R_ind)
{
  /* 每個 codeblock 的 E（rate-matched bits）暫時假設 = G / C */
  if (cfg->C <= 0) {
    fprintf(stderr, "[DERIVE] C=%d 不合理\n", cfg->C);
    return -1;
  }

  if (cfg->G % cfg->C != 0) {
    fprintf(stderr,
            "[DERIVE] G=%d 不能整除 C=%d，暫時還是硬用整除（捨去餘數），請你之後再檢查\n",
            cfg->G, cfg->C);
  }

  int E = cfg->G / cfg->C;
  printf("[DERIVE] assume each CB has E=%d LLRs (G/C)\n", E);

  if (E <= 0) {
    fprintf(stderr, "[DERIVE] E=%d 不合理\n", E);
    return -1;
  }

  /* 近似 code rate：有用 bits (Kprime-24) / E */
  double R_eff = (double)(Kprime - 24) / (double)E;
  printf("[DERIVE] effective code rate ~ (Kprime-24)/E = %.6f\n", R_eff);

  /* OAI testbench 用的 code_rate_vec */
  int code_rate_vec[8] = {15, 13, 25, 12, 23, 34, 56, 89};
  /* 對應的 (nom_rate, denom_rate) 候選組合 */
  double candidates[3];
  int    cand_nom[3];
  int    cand_den[3];

  if (cfg->BG == 1) {
    /* BG1 / K' > 3840: 支援 1/3, 2/3, 22/25 */
    candidates[0] = 1.0/3.0;  cand_nom[0] = 1;  cand_den[0] = 3;
    candidates[1] = 2.0/3.0;  cand_nom[1] = 2;  cand_den[1] = 3;
    candidates[2] = 22.0/25.0; cand_nom[2] = 22; cand_den[2] = 25;
  } else {
    /* BG2: 支援 1/5, 1/3, 2/3 */
    candidates[0] = 1.0/5.0;  cand_nom[0] = 1; cand_den[0] = 5;
    candidates[1] = 1.0/3.0;  cand_nom[1] = 1; cand_den[1] = 3;
    candidates[2] = 2.0/3.0;  cand_nom[2] = 2; cand_den[2] = 3;
  }

  /* 找最接近 R_eff 的一組 */
  double best_diff = 1e9;
  int best_nom = cand_nom[0];
  int best_den = cand_den[0];

  for (int i = 0; i < 3; i++) {
    double diff = fabs(R_eff - candidates[i]);
    if (diff < best_diff) {
      best_diff = diff;
      best_nom = cand_nom[i];
      best_den = cand_den[i];
    }
  }

  printf("[DERIVE] choose code rate approx %d/%d (diff=%.6f)\n", best_nom, best_den, best_diff);

  /* 用原 ldpctest.c 的 mapping 算 R_ind */
  int R_ind = -1;
  int nom_rate  = best_nom;
  int denom_rate = best_den;
  int BG = cfg->BG;
  bool error = false;

  switch (nom_rate) {
    case 1:
      if (denom_rate == 5)
        if (BG == 2)
          R_ind = 0;  /* 1/5, BG2 */
        else
          error = true;
      else if (denom_rate == 3)
        R_ind = 1;    /* 1/3 */
      else if (denom_rate == 2)
        error = true;
      else
        error = true;
      break;

    case 2:
      if (denom_rate == 5)
        error = true;
      else if (denom_rate == 3)
        R_ind = 4;    /* 2/3 */
      else
        error = true;
      break;

    case 22:
      if (denom_rate == 25 && BG == 1)
        R_ind = 7;    /* 22/25, BG1 */
      else
        error = true;
      break;

    default:
      error = true;
  }

  if (error || R_ind < 0) {
    fprintf(stderr,
            "[DERIVE] 無法將 nom_rate=%d denom_rate=%d BG=%d 映射到 R_ind（ldpctest 的規則），請你之後再檢查\n",
            nom_rate, denom_rate, BG);
    return -1;
  }

  printf("[DERIVE] mapped to R_ind=%d → code_rate_vec[R_ind]=%d\n",
         R_ind, code_rate_vec[R_ind]);

  if (out_R_ind)
    *out_R_ind = R_ind;

  return code_rate_vec[R_ind];
}

/* 將 decoder 輸出的 bit-packed bytes 展開成「1 bit → 1 byte 的 0/1」 */
static void unpack_bits_to_bytes(const uint8_t *packed, int n_bits, uint8_t *out_bytes)
{
  for (int i = 0; i < n_bits; i++) {
    uint8_t b = (packed[i / 8] >> (i & 7)) & 0x01;
    out_bytes[i] = b;
  }
}

/* main：ldpctest_spx */
configmodule_interface_t *uniqCfg = NULL;

int main(int argc, char *argv[])
{
  if (argc != 4) {
    fprintf(stderr,
            "Usage: %s <infer_llr_int8.bin> <ldpc_cfg.txt> <decoded_bits.bin>\n",
            argv[0]);
    fprintf(stderr,
            "  注意：cfg 檔為 key-value 文字格式（例如：BG 1, Zc 224, A 9480...）。\n");
    return 1;
  }

  const char *llr_path = argv[1];
  const char *cfg_path = argv[2];
  const char *out_path = argv[3];

  printf("=== ldpctest_spx: SpikingRx → OAI LDPC decoder wrapper ===\n");

  /* 讀 cfg */
  ldpc_cfg_t cfg;
  if (load_ldpc_cfg_kv(cfg_path, &cfg) != 0) {
    fprintf(stderr, "[MAIN] load_ldpc_cfg_kv fail\n");
    return 1;
  }

  /* 讀 LLR */
  int llr_len = 0;
  int8_t *llr_buf = load_llr_file(llr_path, &cfg, &llr_len);
  if (!llr_buf) {
    fprintf(stderr, "[MAIN] load_llr_file fail\n");
    return 1;
  }

  /* 推 Kprime */
  int Kprime = derive_Kprime(&cfg);
  if (Kprime <= 0) {
    fprintf(stderr, "[MAIN] derive_Kprime fail\n");
    free(llr_buf);
    return 1;
  }

  /* 推 E (= 每個 codeblock 的 LLR 數量) */
  if (cfg.C <= 0) {
    fprintf(stderr, "[MAIN] cfg.C=%d 不合理\n", cfg.C);
    free(llr_buf);
    return 1;
  }

  int E = llr_len / cfg.C;
  if (E <= 0) {
    fprintf(stderr,
            "[MAIN] E=llr_len/C = %d 不合理 (llr_len=%d C=%d)\n",
            E, llr_len, cfg.C);
    free(llr_buf);
    return 1;
  }

  if (E > MAX_LLR_LEN) {
    fprintf(stderr,
            "[MAIN] E=%d > MAX_LLR_LEN=%d，需要調整 MAX_LLR_LEN\n",
            E, MAX_LLR_LEN);
    free(llr_buf);
    return 1;
  }

  printf("[MAIN] splitted: C=%d, each CB has E=%d LLRs\n", cfg.C, E);

  /* 選 decParams.R */
  int R_ind = -1;
  int R_val = pick_ldpc_R(&cfg, Kprime, &R_ind);
  if (R_val < 0) {
    fprintf(stderr, "[MAIN] pick_ldpc_R fail\n");
    free(llr_buf);
    return 1;
  }

  /* 初始化 OAI config & log */
  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == 0) {
    exit_fun("[ldpctest_spx] Error, configuration module init failed\n");
  }
  logInit();

  /* 載入 LDPC shared library（跟原 ldpctest.c 相同用法） */
  load_LDPClib("", &ldpc_toCompare);  /* ""：預設版本 */

  /* 建 decoder profile & abort flag */
  t_nrLDPC_time_stats decoder_profiler = {0};
  reset_meas(&decoder_profiler.llr2llrProcBuf);
  reset_meas(&decoder_profiler.llr2CnProcBuf);
  reset_meas(&decoder_profiler.cnProc);
  reset_meas(&decoder_profiler.cnProcPc);
  reset_meas(&decoder_profiler.bnProc);
  reset_meas(&decoder_profiler.bnProcPc);
  reset_meas(&decoder_profiler.cn2bnProcBuf);
  reset_meas(&decoder_profiler.bn2cnProcBuf);
  reset_meas(&decoder_profiler.llrRes2llrOut);
  reset_meas(&decoder_profiler.llr2bit);

  decode_abort_t dec_abort;
  init_abort(&dec_abort);

  /* 建 decParams（每個 segment 一個） */
  if (cfg.C > MAX_SEGMENTS) {
    fprintf(stderr,
            "[MAIN] cfg.C=%d > MAX_SEGMENTS=%d，請調大 MAX_SEGMENTS\n",
            cfg.C, MAX_SEGMENTS);
    free(llr_buf);
    return 1;
  }

  t_nrLDPC_dec_params decParams[MAX_SEGMENTS];
  memset(decParams, 0, sizeof(decParams));

  for (int j = 0; j < cfg.C; j++) {
    decParams[j].BG         = (uint8_t)cfg.BG;
    decParams[j].Z          = (uint16_t)cfg.Zc;
    decParams[j].R          = (uint8_t)R_val;     /* code_rate_vec[R_ind] */
    decParams[j].numMaxIter = 25;                 /* 你可以自己改，預設 25 次 */
    decParams[j].outMode    = nrLDPC_outMode_BIT; /* decoder 輸出 bit-packed 到 byte 陣列 */
    decParams[j].Kprime     = (uint16_t)Kprime;

    printf("[MAIN] decParams[%d]: BG=%d Z=%d R(code_rate)=%d Kprime=%d numMaxIter=%d\n",
           j,
           decParams[j].BG,
           decParams[j].Z,
           decParams[j].R,
           decParams[j].Kprime,
           decParams[j].numMaxIter);
  }

  /* 準備 LLR segment buffer（每個 segment 一個 array） */
  static int8_t llr_seg[MAX_SEGMENTS][MAX_LLR_LEN];

  for (int j = 0; j < cfg.C; j++) {
    memcpy(llr_seg[j], llr_buf + j * E, E);
    /* 如果 decoder 內部預期長度 > E，就只會用到前 E 個；
       剩下的這裡先用 0 補，避免亂值。 */
    if (E < MAX_LLR_LEN) {
      memset(llr_seg[j] + E, 0, MAX_LLR_LEN - E);
    }
  }

  /* 準備 decoder 輸出：bit-packed，大小給到 Kprime bits → Kprime/8 bytes 足夠，
   * 為了簡單直接開 Kprime bytes（一定足以容納 bit-packed），不會出問題。
   */
  uint8_t est_packed[MAX_SEGMENTS][MAX_BLOCK_LENGTH] = {{0}};

  /* 呼叫 LDPCdecoder（一次 init，之後每個 segment decode） */
  ldpc_toCompare.LDPCinit();

  int32_t total_iter = 0;

  for (int j = 0; j < cfg.C; j++) {
    printf("[DEC] Segment %d / %d: calling LDPCdecoder()...\n", j, cfg.C);

    set_abort(&dec_abort, false);

    int32_t n_iter = ldpc_toCompare.LDPCdecoder(&decParams[j],
                                                (int8_t *)llr_seg[j],
                                                (int8_t *)est_packed[j],
                                                &decoder_profiler,
                                                &dec_abort);

    printf("[DEC] Segment %d: LDPCdecoder() returned n_iter = %d\n", j, n_iter);

    if (n_iter < 0) {
      fprintf(stderr,
              "[DEC] LDPCdecoder() failed for segment %d, ret=%d\n",
              j, n_iter);
      free(llr_buf);
      return 1;
    }

    total_iter += n_iter;
  }

  printf("[DEC] average iterations per segment ~ %.2f\n",
         (double)total_iter / (double)cfg.C);

  /* 將所有 segment 的 packed bits 展開成「一 bit 一 byte」的 0/1 */
  int total_bits = cfg.C * Kprime;
  uint8_t *decoded_bits = (uint8_t *)malloc(total_bits);
  if (!decoded_bits) {
    fprintf(stderr, "[OUT] malloc(%d) for decoded_bits fail\n", total_bits);
    free(llr_buf);
    return 1;
  }

  for (int j = 0; j < cfg.C; j++) {
    unpack_bits_to_bytes(est_packed[j],
                         Kprime,
                         decoded_bits + j * Kprime);
  }

  /* 寫出檔案 */
  FILE *fo = fopen(out_path, "wb");
  if (!fo) {
    fprintf(stderr,
            "[OUT] open '%s' fail: %s\n", out_path, strerror(errno));
    free(llr_buf);
    free(decoded_bits);
    return 1;
  }

  size_t nw = fwrite(decoded_bits, 1, total_bits, fo);
  fclose(fo);

  if (nw != (size_t)total_bits) {
    fprintf(stderr,
            "[OUT] fwrite only %zu/%d bytes\n", nw, total_bits);
    free(llr_buf);
    free(decoded_bits);
    return 1;
  }

  printf("[OUT] wrote %d bits as %d bytes (1 bit → 1 byte) to '%s'\n",
         total_bits, total_bits, out_path);

  free(llr_buf);
  free(decoded_bits);

  /* 清理 log/config module */
  loader_reset();
  logTerm();

  return 0;
}

