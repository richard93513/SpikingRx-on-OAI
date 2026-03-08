/*
 * rmunmatch_spx.c  (A13_add_unscrambling_from_pdsch_cfg + init_byte2m128i)
 *
 * Goal:
 *   - Read ldpc_cfg.txt (key-value)
 *   - Read pdsch_cfg.txt (key-value)
 *   - Read UE demapper LLR dump: demapper_llr_f32.bin (float32 container, len=G)
 *   - Reproduce UE path exactly:
 *       demapper LLR (full G)
 *         -> nr_dlsch_unscrambling(G, dlDataScramblingId, rnti)
 *         -> per-segment nr_deinterleaving_ldpc(E, Qm, ...)
 *         -> nr_rate_matching_ldpc_rx(...)
 *         -> UE exact z[] rebuild
 *         -> UE exact simde_mm_packs_epi16() to build decoder input l[]
 *   - Produce outputs that match UE rm_exact dumps:
 *       rm_exact_spx_seg%02d_i8.bin : exactly equals rm_exact_seg%02d_i8.bin
 *
 * Usage:
 *   ./rmunmatch_spx <demapper_llr_f32.bin> <ldpc_cfg.txt> <pdsch_cfg.txt> <rm_exact_spx_prefix> [--llr-scale <float>]
 */

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
#include "executables/nr-uesoftmodem.h"

#include "openair1/PHY/defs_nr_common.h"
#include "PHY/sse_intrin.h"
#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_segment/nr_rate_matching.h"

#ifndef malloc16
#define malloc16(x) memalign(32, x)
#endif

#define MAX_SEGMENTS MAX_NUM_DLSCH_SEGMENTS
#define MAX_OUT_LEN  (68 * 384)   /* BG1 max */

/* ========= Fallback for OAI exit_function (match exact prototype) ========= */
__attribute__((weak))
void exit_function(const char *file, const char *function, const int line,
                   const char *s, const int assert_flag)
{
  fprintf(stderr, "\n[SPX][exit_function] %s:%d (%s): %s (assert=%d)\n",
          file ? file : "?",
          line,
          function ? function : "?",
          s ? s : "?",
          assert_flag);
  fflush(stderr);
  abort();
}

/* ========= Weak stub only to satisfy link from nr_dlsch_decoding.c ========= */
__attribute__((weak))
nrUE_params_t *get_nrUE_params(void)
{
  static nrUE_params_t p;
  memset(&p, 0, sizeof(p));
  return &p;
}

/* ========= Forward declarations ========= */
extern void nr_dlsch_unscrambling(int16_t *llr,
                                  uint32_t size,
                                  uint8_t q,
                                  uint32_t Nid,
                                  uint32_t n_RNTI);

extern void init_byte2m128i(void);

/* ===== ldpc cfg ===== */
typedef struct {
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

  int R; /* optional, not used here */
} ldpc_cfg_t;

/* ===== pdsch cfg ===== */
typedef struct {
  int frame;
  int slot;
  int dlsch_id;
  int harq_pid;
  int rnti;
  int G;
  int dlDataScramblingId;
  int physCellId;
} pdsch_cfg_t;

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

    if      (strcmp(key, "BG")         == 0) cfg->BG         = val;
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
    else if (strcmp(key, "R0")         == 0) cfg->R          = val;
  }

  fclose(f);

  if (cfg->BG == 0 || cfg->Zc == 0 || cfg->A == 0 || cfg->C == 0 ||
      cfg->G == 0 || cfg->K == 0 || cfg->Qm == 0 || cfg->nb_layers == 0) {
    fprintf(stderr,
            "[CFG] invalid ldpc cfg: BG=%d Zc=%d A=%d C=%d K=%d F=%d G=%d Qm=%d L=%d\n",
            cfg->BG, cfg->Zc, cfg->A, cfg->C, cfg->K, cfg->F,
            cfg->G, cfg->Qm, cfg->nb_layers);
    return -1;
  }

  printf("[LDPC_CFG] BG=%d Zc=%d A=%d C=%d K=%d F=%d G=%d Qm=%d nb_layers=%d\n",
         cfg->BG, cfg->Zc, cfg->A, cfg->C, cfg->K, cfg->F, cfg->G,
         cfg->Qm, cfg->nb_layers);
  printf("[LDPC_CFG] rv_index=%d tbslbrm=%d mcs=%d R=%d\n",
         cfg->rv_index, cfg->tbslbrm, cfg->mcs, cfg->R);

  return 0;
}

static int load_pdsch_cfg_kv(const char *path, pdsch_cfg_t *cfg)
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

    if      (strcmp(key, "frame")              == 0) cfg->frame              = val;
    else if (strcmp(key, "slot")               == 0) cfg->slot               = val;
    else if (strcmp(key, "dlsch_id")           == 0) cfg->dlsch_id           = val;
    else if (strcmp(key, "harq_pid")           == 0) cfg->harq_pid           = val;
    else if (strcmp(key, "rnti")               == 0) cfg->rnti               = val;
    else if (strcmp(key, "G")                  == 0) cfg->G                  = val;
    else if (strcmp(key, "dlDataScramblingId") == 0) cfg->dlDataScramblingId = val;
    else if (strcmp(key, "physCellId")         == 0) cfg->physCellId         = val;
  }

  fclose(f);

  if (cfg->rnti < 0 || cfg->G <= 0) {
    fprintf(stderr,
            "[CFG] invalid pdsch cfg: rnti=%d G=%d dlDataScramblingId=%d physCellId=%d\n",
            cfg->rnti, cfg->G, cfg->dlDataScramblingId, cfg->physCellId);
    return -1;
  }

  printf("[PDSCH_CFG] frame=%d slot=%d dlsch_id=%d harq_pid=%d\n",
         cfg->frame, cfg->slot, cfg->dlsch_id, cfg->harq_pid);
  printf("[PDSCH_CFG] rnti=%d G=%d dlDataScramblingId=%d physCellId=%d\n",
         cfg->rnti, cfg->G, cfg->dlDataScramblingId, cfg->physCellId);

  return 0;
}

static uint8_t *read_file_u8(const char *path, long *out_len)
{
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "[IN] open '%s' fail: %s\n", path, strerror(errno));
    return NULL;
  }

  if (fseek(f, 0, SEEK_END) != 0) {
    fprintf(stderr, "[IN] fseek fail for '%s'\n", path);
    fclose(f);
    return NULL;
  }

  long len = ftell(f);
  if (len <= 0) {
    fprintf(stderr, "[IN] bad file len=%ld for '%s'\n", len, path);
    fclose(f);
    return NULL;
  }
  rewind(f);

  uint8_t *buf = (uint8_t *)malloc((size_t)len);
  if (!buf) {
    fprintf(stderr, "[IN] malloc(%ld) fail for '%s'\n", len, path);
    fclose(f);
    return NULL;
  }

  size_t n = fread(buf, 1, (size_t)len, f);
  fclose(f);

  if (n != (size_t)len) {
    fprintf(stderr, "[IN] fread short for '%s': %zu/%ld\n", path, n, len);
    free(buf);
    return NULL;
  }

  *out_len = len;
  return buf;
}

static void print_f32_stats(const float *x, int n, const char *tag)
{
  if (n <= 0)
    return;

  float mn = x[0], mx = x[0];
  double sum = 0.0, sum2 = 0.0;
  int nonfinite = 0;

  for (int i = 0; i < n; i++) {
    float v = x[i];
    if (!isfinite(v)) {
      nonfinite++;
      continue;
    }
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += (double)v;
    sum2 += (double)v * (double)v;
  }

  double mean = sum / (double)n;
  double var  = sum2 / (double)n - mean * mean;
  double std  = (var > 0.0) ? sqrt(var) : 0.0;

  printf("[STAT][%s] n=%d min=%.6f max=%.6f mean=%.6f std=%.6f nonfinite=%d\n",
         tag, n, mn, mx, mean, std, nonfinite);
}

static void print_i16_stats(const int16_t *x, int n, const char *tag)
{
  if (n <= 0)
    return;

  int16_t mn = x[0], mx = x[0];
  double sum = 0.0, sum2 = 0.0;

  for (int i = 0; i < n; i++) {
    const int16_t v = x[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += (double)v;
    sum2 += (double)v * (double)v;
  }

  double mean = sum / (double)n;
  double var  = sum2 / (double)n - mean * mean;
  double std  = (var > 0.0) ? sqrt(var) : 0.0;

  printf("[STAT][%s] n=%d min=%d max=%d mean=%.6f std=%.6f\n",
         tag, n, (int)mn, (int)mx, mean, std);
}

static inline int16_t clamp_i16_from_long(long v)
{
  if (v > 32767L)  return 32767;
  if (v < -32768L) return -32768;
  return (int16_t)v;
}

/* Exact same formula as OAI nr_get_E() */
static uint32_t spx_nr_get_E(uint32_t G, uint8_t C, uint8_t Qm, uint8_t Nl, uint8_t r)
{
  uint32_t E;
  uint8_t Cprime = C; /* assume CBGTI not present */

  AssertFatal(Nl > 0, "Nl is 0\n");
  AssertFatal(Qm > 0, "Qm is 0\n");

  if (r <= Cprime - ((G / (Nl * Qm)) % Cprime) - 1)
    E = Nl * Qm * (G / (Nl * Qm * Cprime));
  else
    E = Nl * Qm * ((G / (Nl * Qm * Cprime)) + 1);

  return E;
}

/* demapper_llr_f32.bin stores llr[0][i] values using float container */
static void convert_f32_container_to_i16(const float *in_f32, int n, float scale, int16_t *out_i16)
{
  if (scale == 1.0f) {
    for (int i = 0; i < n; i++) {
      float v = in_f32[i];
      if (!isfinite(v))
        v = 0.0f;
      out_i16[i] = (int16_t)v; /* direct cast to mirror original integer-valued llr */
    }
    return;
  }

  for (int i = 0; i < n; i++) {
    float v = in_f32[i];
    if (!isfinite(v))
      v = 0.0f;
    long q = (long)(v * scale); /* truncate, do not round */
    out_i16[i] = clamp_i16_from_long(q);
  }
}

/* Use UE exact z[] rebuild + UE exact simde_mm_packs_epi16 path */
static void spx_build_decoder_input_from_d_ue_exact(const ldpc_cfg_t *cfg,
                                                    const int16_t *d_unmatched,
                                                    int8_t *l_out)
{
  const int Z  = cfg->Zc;
  const int K  = cfg->K;
  const int F  = cfg->F;
  const int BG = cfg->BG;
  const int Kc = (BG == 2) ? 52 : 68;
  const int Kprime = K - F;

  int16_t z[MAX_OUT_LEN + 16] __attribute__((aligned(16)));
  memset(z, 0, sizeof(z));

  /* ===== exact copy of UE decoder body =====
   * memset(z, 0, 2*Z)
   * memset(z + Kprime, 127, F)
   * memcpy(z + 2*Z, d, Kprime - 2*Z)
   * memcpy(z + K, d + (K - 2*Z), Kc*Z - K)
   */
  memset(z, 0, (size_t)(2 * Z) * sizeof(*z));
  memset(z + Kprime, 127, (size_t)F * sizeof(*z));
  memcpy(z + 2 * Z,
         d_unmatched,
         (size_t)(Kprime - 2 * Z) * sizeof(*z));
  memcpy(z + K,
         d_unmatched + (K - 2 * Z),
         (size_t)(Kc * Z - K) * sizeof(*z));

  /* ===== exact copy of UE pack path ===== */
  simde__m128i *pv = (simde__m128i *)&z;
  simde__m128i *pl = (simde__m128i *)l_out;

  for (int i = 0, j = 0; j < ((Kc * Z) >> 4) + 1; i += 2, j++) {
    pl[j] = simde_mm_packs_epi16(pv[i], pv[i + 1]);
  }
}

/* main */
configmodule_interface_t *uniqCfg = NULL;

int main(int argc, char *argv[])
{
  printf("=== SPX BUILD TAG: A13_add_unscrambling_from_pdsch_cfg + init_byte2m128i ===\n");

  if (argc < 5) {
    fprintf(stderr,
            "Usage: %s <demapper_llr_f32.bin> <ldpc_cfg.txt> <pdsch_cfg.txt> <rm_exact_spx_prefix> [--llr-scale <float>]\n",
            argv[0]);
    return 1;
  }

  const char *in_llr_path    = argv[1];
  const char *ldpc_cfg_path  = argv[2];
  const char *pdsch_cfg_path = argv[3];
  const char *out_prefix     = argv[4];

  float llr_scale = 1.0f;
  for (int i = 5; i + 1 < argc; i++) {
    if (strcmp(argv[i], "--llr-scale") == 0) {
      llr_scale = (float)atof(argv[i + 1]);
      i++;
    }
  }
  printf("[MAIN] llr_scale=%.6f\n", llr_scale);

  ldpc_cfg_t cfg;
  if (load_ldpc_cfg_kv(ldpc_cfg_path, &cfg) != 0)
    return 1;

  pdsch_cfg_t pdsch_cfg;
  if (load_pdsch_cfg_kv(pdsch_cfg_path, &pdsch_cfg) != 0)
    return 1;

  if (pdsch_cfg.G != cfg.G) {
    fprintf(stderr,
            "[MAIN] G mismatch: ldpc_cfg.G=%d vs pdsch_cfg.G=%d\n",
            cfg.G, pdsch_cfg.G);
    return 1;
  }

  if (cfg.C <= 0 || cfg.C > MAX_SEGMENTS) {
    fprintf(stderr, "[MAIN] bad C=%d\n", cfg.C);
    return 1;
  }

  const int Kc = (cfg.BG == 2) ? 52 : 68;
  const int out_len = Kc * cfg.Zc; /* exact decoder-input int8 length per segment */
  const int Ncb = ((cfg.BG == 2) ? 50 : 66) * cfg.Zc;
  const int Kprime = cfg.K - cfg.F;
  const int Foffset = cfg.K - cfg.F - 2 * cfg.Zc;

  if (out_len <= 0 || out_len > MAX_OUT_LEN || Ncb <= 0 || Ncb > MAX_OUT_LEN) {
    fprintf(stderr, "[MAIN] invalid out_len=%d or Ncb=%d\n", out_len, Ncb);
    return 1;
  }

  if (Kprime <= 0 || Foffset < 0) {
    fprintf(stderr, "[MAIN] invalid Kprime=%d or Foffset=%d\n", Kprime, Foffset);
    return 1;
  }

  printf("[MAIN] out_len=%d (exact decoder-input int8 length/seg)\n", out_len);
  printf("[MAIN] Ncb=%d (rate-unmatched d[] length/seg)\n", Ncb);
  printf("[MAIN] Kprime=%d\n", Kprime);
  printf("[MAIN] Foffset=%d (UE exact: K-F-2Z)\n", Foffset);
  printf("[MAIN] expect demapper LLR count G=%d\n", cfg.G);

  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == 0) {
    exit_fun("[rmunmatch_spx] configuration module init failed\n");
  }
  logInit();
  init_byte2m128i();

  long fbytes = 0;
  uint8_t *raw = read_file_u8(in_llr_path, &fbytes);
  if (!raw) {
    loader_reset();
    logTerm();
    return 1;
  }

  const long expect_bytes = (long)cfg.G * 4L;
  if (fbytes != expect_bytes) {
    fprintf(stderr,
            "[IN] demapper file size mismatch: bytes=%ld expect=%ld (G=%d float32)\n",
            fbytes, expect_bytes, cfg.G);
    free(raw);
    loader_reset();
    logTerm();
    return 1;
  }

  const float *demap_f32 = (const float *)raw;
  print_f32_stats(demap_f32, cfg.G, "demapper_f32");

  int16_t *soft_all_i16 = (int16_t *)malloc16((size_t)cfg.G * sizeof(int16_t));
  if (!soft_all_i16) {
    fprintf(stderr, "[IN] malloc16 soft_all_i16 fail\n");
    free(raw);
    loader_reset();
    logTerm();
    return 1;
  }

  convert_f32_container_to_i16(demap_f32, cfg.G, llr_scale, soft_all_i16);
  free(raw);

  print_i16_stats(soft_all_i16, cfg.G, "pre_unscram_i16");

  /* exact UE position: unscrambling happens before LDPC deinterleaving/rm_rx */
  nr_dlsch_unscrambling(soft_all_i16,
                        (uint32_t)cfg.G,
                        0,
                        (uint32_t)pdsch_cfg.dlDataScramblingId,
                        (uint32_t)pdsch_cfg.rnti);

  print_i16_stats(soft_all_i16, cfg.G, "post_unscram_i16");

  uint32_t r_offset = 0;

  for (int r = 0; r < cfg.C; r++) {
    const uint32_t E = spx_nr_get_E((uint32_t)cfg.G,
                                    (uint8_t)cfg.C,
                                    (uint8_t)cfg.Qm,
                                    (uint8_t)cfg.nb_layers,
                                    (uint8_t)r);

    const uint8_t clear = 1; /* one-shot offline decode: always clear */

    printf("[RM] seg %d/%d: E=%u r_offset=%u F=%d Foffset=%d clear=%u Tbslbrm=%d BG=%d Z=%d rv=%d\n",
           r, cfg.C, E, r_offset, cfg.F, Foffset, clear,
           cfg.tbslbrm, cfg.BG, cfg.Zc, cfg.rv_index);

    if (r_offset + E > (uint32_t)cfg.G) {
      fprintf(stderr, "[RM] segment slice overflow: r_offset=%u E=%u G=%d\n",
              r_offset, E, cfg.G);
      free(soft_all_i16);
      loader_reset();
      logTerm();
      return 1;
    }

    int16_t *soft_in      = (int16_t *)malloc16((size_t)E * sizeof(int16_t));
    int16_t *harq_e       = (int16_t *)malloc16((size_t)E * sizeof(int16_t));
    int16_t *d_unmatched  = (int16_t *)malloc16((size_t)Ncb * sizeof(int16_t));
    int8_t  *l_exact      = (int8_t  *)malloc16((size_t)(out_len + 16));

    if (!soft_in || !harq_e || !d_unmatched || !l_exact) {
      fprintf(stderr, "[RM] malloc fail at seg=%d\n", r);
      if (soft_in) free(soft_in);
      if (harq_e) free(harq_e);
      if (d_unmatched) free(d_unmatched);
      if (l_exact) free(l_exact);
      free(soft_all_i16);
      loader_reset();
      logTerm();
      return 1;
    }

    memcpy(soft_in, soft_all_i16 + r_offset, (size_t)E * sizeof(int16_t));
    memset(harq_e, 0, (size_t)E * sizeof(int16_t));
    memset(d_unmatched, 0, (size_t)Ncb * sizeof(int16_t));
    memset(l_exact, 0, (size_t)(out_len + 16));

    /* exact UE call chain after unscrambling */
    nr_deinterleaving_ldpc(E, (uint8_t)cfg.Qm, harq_e, soft_in);

    int rc = nr_rate_matching_ldpc_rx((uint32_t)cfg.tbslbrm,
                                      (uint8_t)cfg.BG,
                                      (uint16_t)cfg.Zc,
                                      d_unmatched,
                                      harq_e,
                                      (uint8_t)cfg.C,
                                      (uint8_t)cfg.rv_index,
                                      clear,
                                      E,
                                      (uint32_t)cfg.F,
                                      (uint32_t)Foffset);
    if (rc == -1) {
      fprintf(stderr, "[RM] nr_rate_matching_ldpc_rx failed seg=%d rc=%d\n", r, rc);
      free(soft_in);
      free(harq_e);
      free(d_unmatched);
      free(l_exact);
      free(soft_all_i16);
      loader_reset();
      logTerm();
      return 1;
    }

    spx_build_decoder_input_from_d_ue_exact(&cfg, d_unmatched, l_exact);

    char fn[256];
    snprintf(fn, sizeof(fn), "%s_seg%02d_i8.bin", out_prefix, r);

    FILE *fo = fopen(fn, "wb");
    if (!fo) {
      fprintf(stderr, "[OUT] open '%s' fail: %s\n", fn, strerror(errno));
      free(soft_in);
      free(harq_e);
      free(d_unmatched);
      free(l_exact);
      free(soft_all_i16);
      loader_reset();
      logTerm();
      return 1;
    }

    size_t nw = fwrite(l_exact, 1, (size_t)out_len, fo);
    fclose(fo);

    if (nw != (size_t)out_len) {
      fprintf(stderr, "[OUT] short write '%s': %zu/%d\n", fn, nw, out_len);
      free(soft_in);
      free(harq_e);
      free(d_unmatched);
      free(l_exact);
      free(soft_all_i16);
      loader_reset();
      logTerm();
      return 1;
    }

    printf("[OUT] wrote %s (%d bytes)\n", fn, out_len);

    free(soft_in);
    free(harq_e);
    free(d_unmatched);
    free(l_exact);

    r_offset += E;
  }

  free(soft_all_i16);
  loader_reset();
  logTerm();
  return 0;
}
