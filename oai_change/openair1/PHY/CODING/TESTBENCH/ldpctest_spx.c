/*
 * ldpctest_spx.c  (A7.2 CLEAN - rm_exact only, UE-style bytes, SAFE output buf)
 *
 * Goal:
 *   - Read ldpc_cfg.txt (key-value)
 *   - Read UE rm_exact dumps: int8 buffer 'l' that is FED INTO LDPCdecoder() in OAI
 *   - Run ONLY LDPCdecoder() (NO deinterleave, NO rate-unmatching)
 *   - Produce outputs that match UE post-decode dumps in nr_dlsch_decoding.c:
 *       cb_payload_seg%02d.bin : exactly the bytes UE dumps as ue_c_segXX.bin (Kr_bytes each)
 *       tb_payload_bytes.bin  : exactly the bytes UE dumps as ue_tb.bin (A/8 bytes)
 *   - Also produce:
 *       decoded_bits.bin      : A bits, 1bit->1byte (0/1), derived from tb_payload_bytes.bin
 *
 * Usage:
 *   ./ldpctest_spx <rm_exact_input> <ldpc_cfg.txt> <decoded_bits.bin>
 *
 * rm_exact_input can be:
 *   1) concatenated file: size = out_len * C bytes  (seg0 then seg1 ...)
 *   2) seg00 file: size = out_len bytes, e.g. rm_exact_seg00_i8.bin
 *      If C>1, auto-load seg01.. by replacing "seg00" -> "seg01" ...
 *
 * IMPORTANT:
 *   - This tool MUST match OAI segment decoder behavior:
 *       decodeIterations = LDPCdecoder(p_decoderParms, l, llrProcBuf, ...)
 *       if success: memcpy(c, llrProcBuf, K>>3)
 *       UE then uses Kr_bytes = (K>>3)-(F>>3)-((C>1)?3:0)
 *       and memcpy(b+offset, c[r], Kr_bytes)
 *
 *   - DO NOT shrink LDPCdecoder output buffer. OAI uses 27000 bytes to avoid overflow.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <stdbool.h>
#include <malloc.h>

#include "assertions.h"
#include "SIMULATION/TOOLS/sim.h"
#include "common/config/config_userapi.h"
#include "common/utils/load_module_shlib.h"
#include "common/utils/LOG/log.h"
#include "openair1/PHY/defs_nr_common.h"
#include "PHY/CODING/nrLDPC_extern.h"
#include "coding_unitary_defs.h"
#include "openair1/PHY/defs_gNB.h"   // lenWithCrc(), crcType()

#ifndef malloc16
#define malloc16(x) memalign(32, x)
#endif

#define MAX_SEGMENTS     MAX_NUM_DLSCH_SEGMENTS
/* out_len max: BG1 => 68*384=26112 */
#define MAX_OUT_LEN      (68 * 384)

/* MUST be large enough like OAI's segment decoder */
#define OAI_LDPC_DECODER_MAX_NUM_LLR 27000

static ldpc_interface_t ldpc_if;

/* ===== cfg ===== */
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

  int R; /* optional: from ldpc_cfg.txt (R or R0) */
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

  if (cfg->BG == 0 || cfg->Zc == 0 || cfg->A == 0 || cfg->C == 0 || cfg->G == 0 || cfg->K == 0) {
    fprintf(stderr,
            "[CFG] invalid cfg: BG=%d Zc=%d A=%d C=%d K=%d G=%d\n",
            cfg->BG, cfg->Zc, cfg->A, cfg->C, cfg->K, cfg->G);
    return -1;
  }

  printf("[CFG] BG=%d Zc=%d A=%d C=%d K=%d F=%d G=%d Qm=%d nb_layers=%d\n",
         cfg->BG, cfg->Zc, cfg->A, cfg->C, cfg->K, cfg->F, cfg->G,
         cfg->Qm, cfg->nb_layers);
  printf("[CFG] rv_index=%d tbslbrm=%d mcs=%d R=%d\n",
         cfg->rv_index, cfg->tbslbrm, cfg->mcs, cfg->R);

  return 0;
}

static int pick_ldpc_R_from_cfg(const ldpc_cfg_t *cfg)
{
  if (cfg->R > 0) {
    printf("[MAIN] use cfg.R=%d from ldpc_cfg.txt\n", cfg->R);
    return cfg->R;
  }
  fprintf(stderr, "[MAIN][WARN] cfg.R missing, fallback to 13\n");
  return 13;
}

static uint8_t *read_file_u8(const char *path, long *out_len)
{
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "[IN] open '%s' fail: %s\n", path, strerror(errno));
    return NULL;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fprintf(stderr, "[IN] fseek fail\n");
    fclose(f);
    return NULL;
  }
  long len = ftell(f);
  if (len <= 0) {
    fprintf(stderr, "[IN] bad file len=%ld\n", len);
    fclose(f);
    return NULL;
  }
  rewind(f);

  uint8_t *buf = (uint8_t *)malloc((size_t)len);
  if (!buf) {
    fprintf(stderr, "[IN] malloc(%ld) fail\n", len);
    fclose(f);
    return NULL;
  }
  size_t n = fread(buf, 1, (size_t)len, f);
  fclose(f);
  if (n != (size_t)len) {
    fprintf(stderr, "[IN] fread short: %zu/%ld\n", n, len);
    free(buf);
    return NULL;
  }
  *out_len = len;
  return buf;
}

/* build segXX path by replacing "seg00" in a base path */
static int build_seg_path(char *out, size_t out_sz, const char *in_path, int seg)
{
  const char *needle = "seg00";
  const char *pos = strstr(in_path, needle);
  if (!pos) return -1;

  size_t prefix_len = (size_t)(pos - in_path);
  if (prefix_len + 5 + strlen(pos + strlen(needle)) + 1 > out_sz) return -1;

  memcpy(out, in_path, prefix_len);
  snprintf(out + prefix_len, out_sz - prefix_len, "seg%02d%s", seg, pos + strlen(needle));
  return 0;
}

/* Load rm_exact into rm_seg[C][out_len] */
static int load_rm_exact_segments(const char *input_path,
                                 const ldpc_cfg_t *cfg,
                                 int out_len,
                                 int8_t rm_seg[MAX_SEGMENTS][MAX_OUT_LEN])
{
  const int C = cfg->C;
  if (C <= 0 || C > MAX_SEGMENTS) {
    fprintf(stderr, "[IN] bad C=%d\n", C);
    return -1;
  }
  if (out_len <= 0 || out_len > MAX_OUT_LEN) {
    fprintf(stderr, "[IN] bad out_len=%d\n", out_len);
    return -1;
  }

  long fbytes = 0;
  uint8_t *buf = read_file_u8(input_path, &fbytes);
  if (!buf) return -1;

  const long expect_concat = (long)out_len * (long)C;
  const long expect_one    = (long)out_len;

  if (fbytes == expect_concat) {
    printf("[IN] rm_exact: concatenated file: bytes=%ld == out_len*C (%d*%d)\n",
           fbytes, out_len, C);
    for (int r = 0; r < C; r++) {
      memcpy(rm_seg[r], buf + (long)r * out_len, (size_t)out_len);
    }
    free(buf);
    return 0;
  }

  if (fbytes == expect_one) {
    printf("[IN] rm_exact: single-seg file (seg00): bytes=%ld == out_len=%d\n",
           fbytes, out_len);

    memcpy(rm_seg[0], buf, (size_t)out_len);
    free(buf);

    if (C == 1) return 0;

    for (int r = 1; r < C; r++) {
      char seg_path[1024];
      if (build_seg_path(seg_path, sizeof(seg_path), input_path, r) != 0) {
        fprintf(stderr, "[IN] C=%d but input path does not contain 'seg00': %s\n", C, input_path);
        return -1;
      }

      long sb = 0;
      uint8_t *sb_buf = read_file_u8(seg_path, &sb);
      if (!sb_buf) {
        fprintf(stderr, "[IN] missing seg file: %s\n", seg_path);
        return -1;
      }
      if (sb != expect_one) {
        fprintf(stderr, "[IN] seg file size mismatch: %s bytes=%ld expect=%ld\n", seg_path, sb, expect_one);
        free(sb_buf);
        return -1;
      }
      memcpy(rm_seg[r], sb_buf, (size_t)out_len);
      free(sb_buf);
    }

    return 0;
  }

  fprintf(stderr,
          "[IN] unsupported rm_exact file size=%ld. Expect out_len=%d or out_len*C=%ld\n",
          fbytes, out_len, expect_concat);
  free(buf);
  return -1;
}

/* Unpack bytes -> bits (MSB-first) into 0/1 bytes */
static void unpack_bytes_msb_first(const uint8_t *in_bytes, int nbytes, uint8_t *out_bits_1B, int nbits_needed)
{
  int bit_idx = 0;
  for (int i = 0; i < nbytes && bit_idx < nbits_needed; i++) {
    uint8_t b = in_bytes[i];
    for (int k = 7; k >= 0 && bit_idx < nbits_needed; k--) {
      out_bits_1B[bit_idx++] = (b >> k) & 1;
    }
  }
}

/* main */
configmodule_interface_t *uniqCfg = NULL;

int main(int argc, char *argv[])
{
  printf("=== SPX BUILD TAG: A7.2_clean_rm_exact_only (UE-style bytes, SAFE buf) ===\n");

  if (argc != 4) {
    fprintf(stderr, "Usage: %s <rm_exact_input> <ldpc_cfg.txt> <decoded_bits.bin>\n", argv[0]);
    return 1;
  }

  const char *in_path  = argv[1];
  const char *cfg_path = argv[2];
  const char *out_bits_path = argv[3];

  ldpc_cfg_t cfg;
  if (load_ldpc_cfg_kv(cfg_path, &cfg) != 0) return 1;

  if (cfg.C <= 0 || cfg.C > MAX_SEGMENTS) {
    fprintf(stderr, "[MAIN] bad C=%d\n", cfg.C);
    return 1;
  }

  const int out_len = (cfg.BG == 1) ? (68 * cfg.Zc) : (52 * cfg.Zc); // rm_exact bytes per seg
  if (out_len <= 0 || out_len > MAX_OUT_LEN) {
    fprintf(stderr, "[MAIN] out_len=%d invalid\n", out_len);
    return 1;
  }

  const int Kprime_ue = lenWithCrc(cfg.C, cfg.A);
  const int crc_type  = crcType(cfg.C, cfg.A);

  const int K_bytes = cfg.K >> 3;
  const int F_bytes = cfg.F >> 3;
  const int drop_cb_crc_bytes = (cfg.C > 1) ? 3 : 0;
  const int Kr_bytes = K_bytes - F_bytes - drop_cb_crc_bytes;

  const int tb_payload_bytes = cfg.A >> 3;

  printf("[MAIN] out_len=%d (rm_exact bytes/seg)\n", out_len);
  printf("[MAIN] K_bytes=%d F_bytes=%d drop_cb_crc_bytes=%d payload_bytes/seg=%d\n",
         K_bytes, F_bytes, drop_cb_crc_bytes, Kr_bytes);
  printf("[MAIN] tb_payload_bytes=%d (A/8)\n", tb_payload_bytes);
  printf("[MAIN] Kprime_ue=lenWithCrc(C,A)=%d crc_type=%d\n", Kprime_ue, crc_type);

  if (Kr_bytes <= 0) {
    fprintf(stderr, "[MAIN] Kr_bytes invalid: %d\n", Kr_bytes);
    return 1;
  }
  if ((cfg.A & 7) != 0 || (cfg.K & 7) != 0 || (cfg.F & 7) != 0) {
    fprintf(stderr, "[MAIN][WARN] A/K/F not multiple of 8 (A=%d K=%d F=%d). This tool assumes byte-aligned.\n",
            cfg.A, cfg.K, cfg.F);
  }

  // init config/log + LDPC lib
  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == 0) {
    exit_fun("[ldpctest_spx] configuration module init failed\n");
  }
  logInit();
  load_LDPClib("", &ldpc_if);
  ldpc_if.LDPCinit();

  int R_val = pick_ldpc_R_from_cfg(&cfg);

  t_nrLDPC_dec_params decParams[MAX_SEGMENTS];
  memset(decParams, 0, sizeof(decParams));
  for (int j = 0; j < cfg.C; j++) {
    decParams[j].BG         = (uint8_t)cfg.BG;
    decParams[j].Z          = (uint16_t)cfg.Zc;
    decParams[j].R          = (uint8_t)R_val;
    decParams[j].numMaxIter = 25;
    decParams[j].outMode    = 0;                 // MATCH OAI segment decoder
    decParams[j].crc_type   = (uint8_t)crc_type; // MATCH OAI
    decParams[j].Kprime     = (uint16_t)Kprime_ue;
    printf("[MAIN] decParams[%d]: BG=%d Z=%d R=%d Kprime=%d outMode=%d\n",
           j, decParams[j].BG, decParams[j].Z, decParams[j].R, decParams[j].Kprime, decParams[j].outMode);
  }

  // load rm_exact
  static int8_t rm_seg[MAX_SEGMENTS][MAX_OUT_LEN];
  memset(rm_seg, 0, sizeof(rm_seg));
  if (load_rm_exact_segments(in_path, &cfg, out_len, rm_seg) != 0) {
    loader_reset();
    logTerm();
    return 1;
  }

  // decoded c bytes per segment (K_bytes each)
  uint8_t *c_bytes[MAX_SEGMENTS] = {0};
  for (int j = 0; j < cfg.C; j++) {
    c_bytes[j] = (uint8_t *)malloc((size_t)K_bytes);
    if (!c_bytes[j]) {
      fprintf(stderr, "[OUT] malloc c_bytes[%d] fail\n", j);
      for (int t = 0; t < cfg.C; t++) if (c_bytes[t]) free(c_bytes[t]);
      loader_reset();
      logTerm();
      return 1;
    }
    memset(c_bytes[j], 0, (size_t)K_bytes);
  }

  // decode loop (MATCH OAI: output buffer is LARGE (27000), then memcpy first K_bytes)
  t_nrLDPC_time_stats decoder_profiler = {0};
  decode_abort_t dec_abort;
  init_abort(&dec_abort);

  int32_t total_iter = 0;

  for (int j = 0; j < cfg.C; j++) {
    int8_t *llrProcBuf = (int8_t *)memalign(32, (size_t)OAI_LDPC_DECODER_MAX_NUM_LLR);
    if (!llrProcBuf) {
      fprintf(stderr, "[DEC] memalign llrProcBuf fail\n");
      for (int t = 0; t < cfg.C; t++) if (c_bytes[t]) free(c_bytes[t]);
      loader_reset();
      logTerm();
      return 1;
    }
    memset(llrProcBuf, 0, (size_t)OAI_LDPC_DECODER_MAX_NUM_LLR);

    printf("[DEC] seg %d/%d: LDPCdecoder(in rm_exact=%d bytes, outBuf=%d bytes)\n",
           j, cfg.C, out_len, OAI_LDPC_DECODER_MAX_NUM_LLR);

    set_abort(&dec_abort, false);

    int32_t n_iter = ldpc_if.LDPCdecoder(&decParams[j],
                                        (int8_t *)rm_seg[j],
                                        (int8_t *)llrProcBuf,
                                        &decoder_profiler,
                                        &dec_abort);

    printf("[DEC] seg %d: n_iter=%d\n", j, n_iter);
    if (n_iter < 0) {
      fprintf(stderr, "[DEC] LDPCdecoder failed seg=%d ret=%d\n", j, n_iter);
      free(llrProcBuf);
      for (int t = 0; t < cfg.C; t++) if (c_bytes[t]) free(c_bytes[t]);
      loader_reset();
      logTerm();
      return 1;
    }

    // MATCH OAI success criteria: decodeIterations < numMaxIter
    if (n_iter < decParams[j].numMaxIter) {
      memcpy(c_bytes[j], llrProcBuf, (size_t)K_bytes);
    } else {
      memset(c_bytes[j], 0, (size_t)K_bytes);
    }

    total_iter += n_iter;
    free(llrProcBuf);
  }

  printf("[DEC] avg iterations/segment = %.2f\n", (double)total_iter / (double)cfg.C);

  // dump UE-style CB payload bytes per seg: first Kr_bytes of c_bytes[j]
  for (int j = 0; j < cfg.C; j++) {
    char fn[64];
    snprintf(fn, sizeof(fn), "cb_payload_seg%02d.bin", j);
    FILE *fc = fopen(fn, "wb");
    if (!fc) {
      fprintf(stderr, "[DUMP] cannot open %s (%s)\n", fn, strerror(errno));
      continue;
    }
    size_t nw = fwrite((void *)c_bytes[j], 1, (size_t)Kr_bytes, fc);
    fclose(fc);
    if (nw != (size_t)Kr_bytes) {
      fprintf(stderr, "[DUMP] short write %s: %zu/%d\n", fn, nw, Kr_bytes);
    } else {
      printf("[DUMP] wrote %s (%d bytes)\n", fn, Kr_bytes);
    }
  }

  // dump UE-style TB bytes: concat all segments' cb payload bytes
  uint8_t *tb_bytes = (uint8_t *)malloc((size_t)tb_payload_bytes);
  if (!tb_bytes) {
    fprintf(stderr, "[TB] malloc tb_bytes fail\n");
    for (int t = 0; t < cfg.C; t++) if (c_bytes[t]) free(c_bytes[t]);
    loader_reset();
    logTerm();
    return 1;
  }
  memset(tb_bytes, 0, (size_t)tb_payload_bytes);

  int off = 0;
  for (int j = 0; j < cfg.C; j++) {
    const int remain = tb_payload_bytes - off;
    if (remain <= 0)
      break;

    const int copy_len = (Kr_bytes < remain) ? Kr_bytes : remain;
    memcpy(tb_bytes + off, c_bytes[j], (size_t)copy_len);
    off += copy_len;
  }

  if (off != tb_payload_bytes) {
    fprintf(stderr, "[TB][WARN] TB bytes length mismatch: built=%d expect=%d\n", off, tb_payload_bytes);
  }

  {
    FILE *ft = fopen("tb_payload_bytes.bin", "wb");
    if (!ft) {
      fprintf(stderr, "[DUMP] cannot open tb_payload_bytes.bin (%s)\n", strerror(errno));
    } else {
      size_t nw = fwrite(tb_bytes, 1, (size_t)tb_payload_bytes, ft);
      fclose(ft);
      if (nw != (size_t)tb_payload_bytes) {
        fprintf(stderr, "[DUMP] short write tb_payload_bytes.bin: %zu/%d\n", nw, tb_payload_bytes);
      } else {
        printf("[DUMP] wrote tb_payload_bytes.bin (%d bytes)\n", tb_payload_bytes);
      }
    }
  }

  // decoded_bits.bin: A bits, 1bit->1byte (0/1), derived from tb_payload_bytes.bin
  {
    uint8_t *bits_1B = (uint8_t *)malloc((size_t)cfg.A);
    if (!bits_1B) {
      fprintf(stderr, "[OUT] malloc bits_1B fail\n");
    } else {
      unpack_bytes_msb_first(tb_bytes, tb_payload_bytes, bits_1B, cfg.A);
      FILE *fo = fopen(out_bits_path, "wb");
      if (!fo) {
        fprintf(stderr, "[OUT] open '%s' fail: %s\n", out_bits_path, strerror(errno));
      } else {
        size_t nw = fwrite(bits_1B, 1, (size_t)cfg.A, fo);
        fclose(fo);
        if (nw != (size_t)cfg.A) {
          fprintf(stderr, "[OUT] short write '%s': %zu/%d\n", out_bits_path, nw, cfg.A);
        } else {
          printf("[OUT] wrote decoded bits A=%d (1bit->1byte) to '%s'\n", cfg.A, out_bits_path);
        }
      }
      free(bits_1B);
    }
  }

  // free
  free(tb_bytes);
  for (int j = 0; j < cfg.C; j++) if (c_bytes[j]) free(c_bytes[j]);

  loader_reset();
  logTerm();
  return 0;
}
