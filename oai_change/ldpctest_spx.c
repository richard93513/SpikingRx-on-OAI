// openair1/PHY/CODING/TESTBENCH/ldpctest_spx.c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <stdbool.h>
#include <math.h>
#include <malloc.h>

#include "assertions.h"
#include "SIMULATION/TOOLS/sim.h"
#include "common/config/config_userapi.h"
#include "common/utils/load_module_shlib.h"
#include "common/utils/LOG/log.h"

#include "PHY/sse_intrin.h" // simde
#include "nr_rate_matching.h" // nr_deinterleaving_ldpc + nr_rate_matching_ldpc_rx declarations

#include "PHY/CODING/coding_extern.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/nrLDPC_extern.h"   // LDPCdecoder prototype + t_nrLDPC_dec_params
#include "coding_unitary_defs.h"

#ifndef malloc16
#define malloc16(x) memalign(32, x)
#endif

// Same style as segment decoder
#define OAI_LDPC_DECODER_MAX_NUM_LLR 27000

// For safety, BG1: 68*384 = 26112
#define MAX_CB_BITS_BG1   (68 * 384)
#define MAX_CB_BITS_BG2   (52 * 384)
#define MAX_CB_BITS       (68 * 384)
#define MAX_SEGMENTS      (MAX_NUM_DLSCH_SEGMENTS)

// ------------------------------------------------------------
// Simple key-value cfg loader (your existing format)
// ------------------------------------------------------------
typedef struct {
  int frame;
  int slot;
  int dlsch_id;

  int BG;         // 1 or 2
  int Zc;         // lifting size
  int A;          // TB payload bits (no TB-CRC)
  int C;          // number of codeblocks
  int K;          // codeblock size at decoder output (OAI uses 22Z for BG1, 10Z for BG2)
  int F;          // filler bits total (for this TB)
  int G;          // total number of LLRs (rate-matched bits)
  int Qm;
  int nb_layers;
  int rv_index;
  int tbslbrm;
  int mcs;

  // IMPORTANT: you must provide R used by OAI segment parameters
  // (segment_decoder uses nrLDPC_TB_decoding_parameters->segments[r].R)
  int R; // OAI code_rate_vec element, e.g. 23 for ~2/3 in your case
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
    else if (strcmp(key, "R")          == 0) cfg->R          = val;
  }

  fclose(f);

  if (cfg->BG == 0 || cfg->Zc == 0 || cfg->A == 0 || cfg->C == 0 || cfg->G == 0 || cfg->K == 0) {
    fprintf(stderr,
            "[CFG] invalid cfg: BG=%d Zc=%d A=%d C=%d G=%d K=%d\n",
            cfg->BG, cfg->Zc, cfg->A, cfg->C, cfg->G, cfg->K);
    return -1;
  }

  if (cfg->R == 0) {
    fprintf(stderr,
            "[CFG] missing R in cfg. You MUST write a line like: R 23\n"
            "      (this is the OAI code_rate_vec element used in segment params)\n");
    return -1;
  }

  printf("[CFG] frame=%d slot=%d dlsch_id=%d\n", cfg->frame, cfg->slot, cfg->dlsch_id);
  printf("[CFG] BG=%d Zc=%d A=%d C=%d K=%d F=%d G=%d Qm=%d nb_layers=%d\n",
         cfg->BG, cfg->Zc, cfg->A, cfg->C, cfg->K, cfg->F, cfg->G, cfg->Qm, cfg->nb_layers);
  printf("[CFG] rv_index=%d tbslbrm=%d mcs=%d R=%d\n",
         cfg->rv_index, cfg->tbslbrm, cfg->mcs, cfg->R);

  return 0;
}

static int8_t *load_llr_file(const char *path, int expected_len, int *out_len)
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

  printf("[LLR] file bytes = %ld (expect G=%d)\n", len, expected_len);
  if (expected_len > 0 && len != expected_len) {
    fprintf(stderr, "[LLR] WARNING: file length (%ld) != expected G (%d)\n", len, expected_len);
  }

  int8_t *buf = (int8_t *)malloc((size_t)len);
  if (!buf) {
    fprintf(stderr, "[LLR] malloc(%ld) fail\n", len);
    fclose(f);
    return NULL;
  }

  size_t n = fread(buf, 1, (size_t)len, f);
  fclose(f);

  if (n != (size_t)len) {
    fprintf(stderr, "[LLR] fread only %zu/%ld bytes\n", n, len);
    free(buf);
    return NULL;
  }

  *out_len = (int)len;
  return buf;
}

static void unpack_bits_to_bytes(const uint8_t *packed, int n_bits, uint8_t *out_bytes)
{
  for (int i = 0; i < n_bits; i++) {
    uint8_t b = (packed[i / 8] >> (i & 7)) & 0x01;
    out_bytes[i] = b;
  }
}

// Reconstruct TB payload bits (A bits) from CB bits (each CB length K bits)
// Logic matches 38.212 segmentation:
// - B  = A + 24 (TB-CRC)
// - B' = B + 24*C (CB-CRC per CB)
// - Kplus = ceil(B'/C), Kminus=floor(B'/C), Cminus=C*Kplus - B'
// - For each CB r: Kr = (r < Cminus)?Kminus:Kplus
// - Filler in CB = Fr = K - Kr  (K is 22Z or 10Z; Kr is actual data length incl CB-CRC)
// - Drop first Fr bits (filler), drop last 24 bits (CB-CRC), append remaining (Kr-24) bits into TB
static int reconstruct_tb_payload_bits(const ldpc_cfg_t *cfg,
                                       const uint8_t *cb_bits, // length C*K (1 bit -> 1 byte)
                                       uint8_t *tb_payload_out) // length A
{
  const int C = cfg->C;
  const int A = cfg->A;
  const int K = cfg->K;

  const int B  = A + 24;
  const int Bp = B + 24 * C;

  const int Kplus  = (Bp + C - 1) / C;
  const int Kminus = Bp / C;
  const int Cminus = C * Kplus - Bp;

  printf("[TB] B=%d B'=%d, Kplus=%d Kminus=%d Cminus=%d\n",
         B, Bp, Kplus, Kminus, Cminus);

  uint8_t *tb_with_crc = (uint8_t *)malloc((size_t)B);
  if (!tb_with_crc) {
    fprintf(stderr, "[TB] malloc(B=%d) fail\n", B);
    return -1;
  }

  int tb_idx = 0;

  for (int r = 0; r < C; r++) {
    const int Kr = (r < Cminus) ? Kminus : Kplus; // includes CB-CRC
    const int Fr = K - Kr;                        // filler bits count in this CB

    if (Fr < 0) {
      fprintf(stderr, "[TB] negative filler: r=%d Fr=%d (K=%d Kr=%d)\n", r, Fr, K, Kr);
      free(tb_with_crc);
      return -1;
    }
    if (Kr <= 24) {
      fprintf(stderr, "[TB] Kr<=24 invalid: r=%d Kr=%d\n", r, Kr);
      free(tb_with_crc);
      return -1;
    }

    const uint8_t *cb = cb_bits + r * K;

    const uint8_t *info_with_cbcrc = cb + Fr;   // skip filler
    const int payload_len = Kr - 24;            // drop CB-CRC (last 24 bits)

    for (int i = 0; i < payload_len; i++) {
      if (tb_idx >= B) {
        fprintf(stderr, "[TB] overflow tb_idx=%d B=%d\n", tb_idx, B);
        free(tb_with_crc);
        return -1;
      }
      tb_with_crc[tb_idx++] = info_with_cbcrc[i];
    }
  }

  if (tb_idx != B) {
    fprintf(stderr, "[TB] assembled TB bits mismatch: expect B=%d, got %d\n", B, tb_idx);
    free(tb_with_crc);
    return -1;
  }

  // Output only A payload bits (drop TB-CRC)
  memcpy(tb_payload_out, tb_with_crc, (size_t)A);
  free(tb_with_crc);

  printf("[TB] reconstructed TB payload bits = %d (A)\n", A);
  return 0;
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
configmodule_interface_t *uniqCfg = NULL;

int main(int argc, char *argv[])
{
  if (argc != 4) {
    fprintf(stderr,
            "Usage: %s <infer_llr_int8.bin> <ldpc_cfg.txt> <decoded_bits.bin>\n",
            argv[0]);
    fprintf(stderr,
            "  cfg 為 key-value 文字檔，且必須包含 R（例如：R 23）。\n");
    return 1;
  }

  const char *llr_path = argv[1];
  const char *cfg_path = argv[2];
  const char *out_path = argv[3];

  printf("=== ldpctest_spx: OAI-segment-decoder-equivalent LDPC decode ===\n");

  // Load cfg
  ldpc_cfg_t cfg;
  if (load_ldpc_cfg_kv(cfg_path, &cfg) != 0) {
    fprintf(stderr, "[MAIN] load_ldpc_cfg_kv fail\n");
    return 1;
  }

  // Load LLR int8 file
  int llr_len = 0;
  int8_t *llr_buf = load_llr_file(llr_path, cfg.G, &llr_len);
  if (!llr_buf) {
    fprintf(stderr, "[MAIN] load_llr_file fail\n");
    return 1;
  }

  if (cfg.C <= 0) {
    fprintf(stderr, "[MAIN] invalid C=%d\n", cfg.C);
    free(llr_buf);
    return 1;
  }

  if (llr_len % cfg.C != 0) {
    fprintf(stderr, "[MAIN] WARNING: llr_len=%d not divisible by C=%d. Using floor.\n", llr_len, cfg.C);
  }

  const int E = llr_len / cfg.C;
  printf("[MAIN] split: C=%d, each segment E=%d LLRs\n", cfg.C, E);

  // Init OAI config/log (required by some OAI utilities)
  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == 0) {
    exit_fun("[ldpctest_spx] Error, configuration module init failed\n");
  }
  logInit();

  // Decoder params: replicate segment_decoder usage
  t_nrLDPC_dec_params decParams = {0};
  decParams.check_crc = check_crc;
  decParams.BG = (uint8_t)cfg.BG;
  decParams.Z  = (uint16_t)cfg.Zc;
  decParams.R  = 23;       // MUST come from cfg (OAI segment param)
  decParams.numMaxIter = 25;
  decParams.outMode = 0;               // EXACTLY like segment_decoder (decParams.outMode = 0)

  // Compute Kc as in segment_decoder
  const uint8_t Kc = (decParams.BG == 2) ? 52 : 68;

  const int K = cfg.K;
  const int Z = cfg.Zc;

  // Basic sanity
  if (K <= 0 || Z <= 0) {
    fprintf(stderr, "[MAIN] invalid K=%d Z=%d\n", K, Z);
    free(llr_buf);
    logTerm();
    return 1;
  }
  if (Kc * Z > MAX_CB_BITS) {
    fprintf(stderr, "[MAIN] Kc*Z=%d exceeds MAX_CB_BITS=%d\n", Kc * Z, MAX_CB_BITS);
    free(llr_buf);
    logTerm();
    return 1;
  }

  // Allocate per segment buffers and outputs
  // - segment_decoder uses "short *llr" and "int16_t harq_e[E]" on stack
  // We'll allocate dynamically to avoid huge stack.
  short  *ulsch_llr = (short *)malloc((size_t)E * sizeof(short));
  int16_t *harq_e   = (int16_t *)malloc((size_t)E * sizeof(int16_t));
  if (!ulsch_llr || !harq_e) {
    fprintf(stderr, "[MAIN] malloc for ulsch_llr/harq_e failed\n");
    free(llr_buf);
    free(ulsch_llr);
    free(harq_e);
    logTerm();
    return 1;
  }

  // Ncb calculation is internal to nr_rate_matching_ldpc_rx, but we need buffer d[]
  // We can compute Ncb here exactly like nr_rate_matching does:
  const uint32_t N = (cfg.BG == 1) ? (66 * (uint32_t)Z) : (50 * (uint32_t)Z);
  uint32_t Ncb;
  if (cfg.tbslbrm == 0) {
    Ncb = N;
  } else {
    // R_LBRM = 2/3 => Nref = 3*Tbslbrm/(2*C)
    uint32_t Nref = (3 * (uint32_t)cfg.tbslbrm) / (2 * (uint32_t)cfg.C);
    Ncb = (N < Nref) ? N : Nref;
  }
  printf("[MAIN] BG=%d Z=%d => N=%u, tbslbrm=%d => Ncb=%u\n", cfg.BG, Z, N, cfg.tbslbrm, Ncb);

  // d buffer per segment (for soft combining)
  int16_t *d_buf[MAX_SEGMENTS] = {0};
  bool d_to_be_cleared[MAX_SEGMENTS] = {0};

  if (cfg.C > MAX_SEGMENTS) {
    fprintf(stderr, "[MAIN] C=%d exceeds MAX_SEGMENTS=%d\n", cfg.C, MAX_SEGMENTS);
    free(llr_buf);
    free(ulsch_llr);
    free(harq_e);
    logTerm();
    return 1;
  }

  for (int r = 0; r < cfg.C; r++) {
    d_buf[r] = (int16_t *)malloc((size_t)Ncb * sizeof(int16_t));
    if (!d_buf[r]) {
      fprintf(stderr, "[MAIN] malloc d_buf[%d] Ncb=%u failed\n", r, Ncb);
      for (int k = 0; k < r; k++) free(d_buf[k]);
      free(llr_buf);
      free(ulsch_llr);
      free(harq_e);
      logTerm();
      return 1;
    }
    d_to_be_cleared[r] = true; // first time clear
  }

  // Output packed CB bits: each CB output is K bits => K/8 bytes
  const int cb_packed_bytes = (K >> 3);
  uint8_t *c_packed[MAX_SEGMENTS] = {0};
  for (int r = 0; r < cfg.C; r++) {
    c_packed[r] = (uint8_t *)malloc((size_t)cb_packed_bytes);
    if (!c_packed[r]) {
      fprintf(stderr, "[MAIN] malloc c_packed[%d] failed\n", r);
      for (int k = 0; k < cfg.C; k++) {
        if (d_buf[k]) free(d_buf[k]);
        if (c_packed[k]) free(c_packed[k]);
      }
      free(llr_buf);
      free(ulsch_llr);
      free(harq_e);
      logTerm();
      return 1;
    }
    memset(c_packed[r], 0, (size_t)cb_packed_bytes);
  }

  // Timing structs (minimal)
  time_stats_t ts_deinterleave = {0};
  time_stats_t ts_rate_unmatch = {0};
  time_stats_t ts_ldpc_decode  = {0};
  reset_meas(&ts_deinterleave);
  reset_meas(&ts_rate_unmatch);
  reset_meas(&ts_ldpc_decode);

  t_nrLDPC_time_stats procTime = {0};
  t_nrLDPC_time_stats *p_procTime = &procTime;

  decode_abort_t dec_abort;
  init_abort(&dec_abort);

  // --------------------------
  // Per-segment decode (clone nr_process_decode_segment)
  // --------------------------
  for (int r = 0; r < cfg.C; r++) {
    printf("[DEC] segment %d/%d\n", r, cfg.C);

    // ulsch_llr (short) is the input to nr_deinterleaving_ldpc
    // fill from int8 file slice
    const int8_t *src = llr_buf + r * E;
    for (int i = 0; i < E; i++) {
      ulsch_llr[i] = (short)src[i];
    }

    // --- deinterleaving ---
    start_meas(&ts_deinterleave);
    nr_deinterleaving_ldpc((uint32_t)E, (uint8_t)cfg.Qm, harq_e, (int16_t *)ulsch_llr);
    stop_meas(&ts_deinterleave);

    // --- rate unmatching ---
    start_meas(&ts_rate_unmatch);

    // EXACTLY the same Foffset expression from segment_decoder:
    // Foffset = K - F - 2*Z
    const uint32_t Foffset = (uint32_t)(K - cfg.F - 2 * Z);

    int ret = nr_rate_matching_ldpc_rx((uint32_t)cfg.tbslbrm,
                                       (uint8_t)cfg.BG,
                                       (uint16_t)Z,
                                       d_buf[r],
                                       harq_e,
                                       (uint8_t)cfg.C,
                                       (uint8_t)cfg.rv_index,
                                       d_to_be_cleared[r] ? 1 : 0,
                                       (uint32_t)E,
                                       (uint32_t)cfg.F,
                                       (uint32_t)Foffset);
    stop_meas(&ts_rate_unmatch);

    if (ret == -1) {
      fprintf(stderr, "[DEC] nr_rate_matching_ldpc_rx failed on segment %d\n", r);
      goto cleanup_fail;
    }
    d_to_be_cleared[r] = false;

    // Set crc_type / Kprime EXACTLY like segment_decoder
    // NOTE: these functions are from OAI coding helpers
    decParams.crc_type = crcType((uint32_t)cfg.C, (uint32_t)cfg.A);
    decParams.Kprime   = (uint16_t)lenWithCrc((uint32_t)cfg.C, (uint32_t)cfg.A);

    // --- build z[] and saturate to int8 l[] EXACTLY like segment_decoder ---
    int16_t zbuf[MAX_CB_BITS + 16] __attribute__((aligned(16)));
    int8_t  lbuf[MAX_CB_BITS + 16] __attribute__((aligned(16)));
    memset(zbuf, 0, sizeof(zbuf));
    memset(lbuf, 0, sizeof(lbuf));

    start_meas(&ts_ldpc_decode);

    // local variables exactly like segment_decoder
    const int Kprime = K - cfg.F;

    // set first 2*Z bits to zero
    memset(zbuf, 0, (size_t)(2 * Z) * sizeof(*zbuf));

    // set filler bits (127) at z + Kprime, length F
    memset(zbuf + Kprime, 127, (size_t)cfg.F * sizeof(*zbuf));

    // Move coded bits before filler bits:
    // memcpy(z + 2Z, d, (Kprime - 2Z))
    if (Kprime < 2 * Z) {
      fprintf(stderr, "[DEC] invalid: Kprime=%d < 2Z=%d\n", Kprime, 2 * Z);
      stop_meas(&ts_ldpc_decode);
      goto cleanup_fail;
    }
    memcpy(zbuf + 2 * Z, d_buf[r], (size_t)(Kprime - 2 * Z) * sizeof(*zbuf));

    // skip filler bits:
    // memcpy(z + K, d + (K - 2Z), (Kc*Z - K))
    if (Kc * Z < K) {
      fprintf(stderr, "[DEC] invalid: Kc*Z=%d < K=%d\n", Kc * Z, K);
      stop_meas(&ts_ldpc_decode);
      goto cleanup_fail;
    }
    memcpy(zbuf + K,
           d_buf[r] + (K - 2 * Z),
           (size_t)(Kc * Z - K) * sizeof(*zbuf));

    // Saturate coded bits into 8-bit values
    simde__m128i *pv = (simde__m128i *)&zbuf;
    simde__m128i *pl = (simde__m128i *)&lbuf;

    // Same loop form as segment_decoder
    for (int i = 0, j = 0; j < ((Kc * Z) >> 4) + 1; i += 2, j++) {
      pl[j] = simde_mm_packs_epi16(pv[i], pv[i + 1]);
    }

    // --- LDPCdecoder (DIRECT call, like segment_decoder) ---
    int8_t llrProcBuf[OAI_LDPC_DECODER_MAX_NUM_LLR] __attribute__((aligned(32)));
    memset(llrProcBuf, 0, sizeof(llrProcBuf));

    set_abort(&dec_abort, false);

    int decodeIterations = LDPCdecoder(&decParams, lbuf, llrProcBuf, p_procTime, &dec_abort);

    if (decodeIterations < decParams.numMaxIter) {
      memcpy(c_packed[r], llrProcBuf, (size_t)cb_packed_bytes);
      printf("[DEC] segment %d: success, iters=%d\n", r, decodeIterations);
    } else {
      memset(c_packed[r], 0, (size_t)cb_packed_bytes);
      printf("[DEC] segment %d: FAIL (hit max iters=%d)\n", r, decodeIterations);
    }

    stop_meas(&ts_ldpc_decode);
  }

  // --------------------------
  // Collect CB packed bits => unpack => TB payload reconstruction
  // --------------------------
  const int total_cb_bits = cfg.C * cfg.K;
  uint8_t *decoded_cb_bits = (uint8_t *)malloc((size_t)total_cb_bits);
  if (!decoded_cb_bits) {
    fprintf(stderr, "[OUT] malloc decoded_cb_bits (%d) fail\n", total_cb_bits);
    goto cleanup_fail;
  }

  for (int r = 0; r < cfg.C; r++) {
    unpack_bits_to_bytes(c_packed[r], cfg.K, decoded_cb_bits + r * cfg.K);
  }

  uint8_t *tb_payload = (uint8_t *)malloc((size_t)cfg.A);
  if (!tb_payload) {
    fprintf(stderr, "[OUT] malloc tb_payload (A=%d) fail\n", cfg.A);
    free(decoded_cb_bits);
    goto cleanup_fail;
  }

  if (reconstruct_tb_payload_bits(&cfg, decoded_cb_bits, tb_payload) != 0) {
    fprintf(stderr, "[OUT] reconstruct TB failed\n");
    free(decoded_cb_bits);
    free(tb_payload);
    goto cleanup_fail;
  }

  // Write A bits, 1 bit -> 1 byte (0/1)
  FILE *fo = fopen(out_path, "wb");
  if (!fo) {
    fprintf(stderr, "[OUT] open '%s' fail: %s\n", out_path, strerror(errno));
    free(decoded_cb_bits);
    free(tb_payload);
    goto cleanup_fail;
  }

  size_t nw = fwrite(tb_payload, 1, (size_t)cfg.A, fo);
  fclose(fo);

  if (nw != (size_t)cfg.A) {
    fprintf(stderr, "[OUT] fwrite only %zu/%d bytes\n", nw, cfg.A);
    free(decoded_cb_bits);
    free(tb_payload);
    goto cleanup_fail;
  }

  printf("[OUT] wrote TB payload A=%d bits (1 bit -> 1 byte) to '%s'\n", cfg.A, out_path);

  free(decoded_cb_bits);
  free(tb_payload);

  // Cleanup
  free(llr_buf);
  free(ulsch_llr);
  free(harq_e);
  for (int r = 0; r < cfg.C; r++) {
    free(d_buf[r]);
    free(c_packed[r]);
  }

  loader_reset();
  logTerm();
  return 0;

cleanup_fail:
  free(llr_buf);
  free(ulsch_llr);
  free(harq_e);
  for (int r = 0; r < cfg.C; r++) {
    if (d_buf[r]) free(d_buf[r]);
    if (c_packed[r]) free(c_packed[r]);
  }
  loader_reset();
  logTerm();
  return 1;
}

