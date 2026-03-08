/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

/*! \file PHY/CODING/nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_decoder.c
 * \brief Top-level routines for decoding LDPC transport channels
 */

#include "nr_rate_matching.h"
#include "PHY/defs_gNB.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"
#include "PHY/CODING/nrLDPC_extern.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_TRANSPORT/nr_ulsch.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "SCHED_NR/sched_nr.h"
#include "defs.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "common/utils/LOG/log.h"

#include <stdalign.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

// simde types used for packing (matches typical OAI usage)
#include <simde/x86/sse2.h>

// #define gNB_DEBUG_TRACE

#define OAI_LDPC_DECODER_MAX_NUM_LLR 27000 // >= max needed

// ===== SPX dump control =====
static int spx_dump_en = 1;
static const char *spx_raw_dir = "/home/richard93513/SpikingRx-on-OAI/spx_records/raw";

static inline void spx_mkdir_p(const char *path, mode_t mode)
{
  if (mkdir(path, mode) != 0 && errno != EEXIST) {
    LOG_W(PHY, "[SPX] mkdir(%s) failed: %s\n", path, strerror(errno));
  }
}

static inline void spx_prepare_dump_dirs(void)
{
  spx_mkdir_p("/home/richard93513/SpikingRx-on-OAI", 0755);
  spx_mkdir_p("/home/richard93513/SpikingRx-on-OAI/spx_records", 0755);
  spx_mkdir_p(spx_raw_dir, 0755);
}

static inline void spx_dump_i16_file(const char *fn, const int16_t *buf, int n_elem)
{
  FILE *fp = fopen(fn, "wb");
  if (!fp) {
    LOG_W(PHY, "[SPX] cannot open i16 dump: %s (%s)\n", fn, strerror(errno));
    return;
  }

  const size_t nwrite = fwrite((const void *)buf, sizeof(int16_t), (size_t)n_elem, fp);
  fclose(fp);

  if (nwrite != (size_t)n_elem) {
    LOG_W(PHY, "[SPX] i16 short write: expect=%d wrote=%zu -> %s\n", n_elem, nwrite, fn);
  }
}

static inline void spx_dump_i8_file(const char *fn, const int8_t *buf, int n_elem)
{
  FILE *fp = fopen(fn, "wb");
  if (!fp) {
    LOG_W(PHY, "[SPX] cannot open i8 dump: %s (%s)\n", fn, strerror(errno));
    return;
  }

  const size_t nwrite = fwrite((const void *)buf, sizeof(int8_t), (size_t)n_elem, fp);
  fclose(fp);

  if (nwrite != (size_t)n_elem) {
    LOG_W(PHY, "[SPX] i8 short write: expect=%d wrote=%zu -> %s\n", n_elem, nwrite, fn);
  }
}

// ===== SPX: deterministic per-(frame,slot) index (same scheme as other dumps) =====
static inline uint32_t spx_make_idx(int frame, int slot)
{
  return (uint32_t)frame * 100u + (uint32_t)slot;
}
// ===============================================================================

#ifdef DEBUG_CRC
#define PRINT_CRC_CHECK(a) a
#else
#define PRINT_CRC_CHECK(a)
#endif

#include "nfapi/open-nFAPI/nfapi/public_inc/nfapi_interface.h"
#include "nfapi/open-nFAPI/nfapi/public_inc/nfapi_nr_interface.h"

/**
 * \typedef nrLDPC_decoding_parameters_t
 * \struct nrLDPC_decoding_parameters_s
 * \brief decoding parameter of transport blocks
 */
typedef struct nrLDPC_decoding_parameters_s {
  t_nrLDPC_dec_params decoderParms;

  uint8_t Qm;

  uint8_t Kc;
  uint8_t rv_index;
  decode_abort_t *abort_decode;

  uint32_t tbslbrm;
  uint32_t A;
  uint32_t K;
  uint32_t Z;
  uint32_t F;

  uint32_t C;

  int E;
  short *llr;
  int16_t *d;
  bool *d_to_be_cleared;
  uint8_t *c;
  bool *decodeSuccess;

  task_ans_t *ans;

  time_stats_t *p_ts_deinterleave;
  time_stats_t *p_ts_rate_unmatch;
  time_stats_t *p_ts_ldpc_decode;

  // ===== SPX: identity for stable dump naming (align across pipeline) =====
  uint16_t spx_frame;
  uint8_t  spx_slot;
  uint32_t spx_idx;
  uint8_t  spx_tb; // TB index within slot (pusch_id) -> prevents filename collisions
  uint8_t  r;      // segment index within TB (0..C-1)
} nrLDPC_decoding_parameters_t;

static void nr_process_decode_segment(void *arg)
{
  nrLDPC_decoding_parameters_t *rdata = (nrLDPC_decoding_parameters_t *)arg;
  t_nrLDPC_dec_params *p_decoderParms = &rdata->decoderParms;
  const int K = rdata->K;
  const int Kprime = K - rdata->F;
  const int A = rdata->A;
  const int E = rdata->E;
  const int Qm = rdata->Qm;
  const int rv_index = rdata->rv_index;
  const uint8_t Kc = rdata->Kc;
  short *ulsch_llr = rdata->llr;

  int8_t llrProcBuf[OAI_LDPC_DECODER_MAX_NUM_LLR] __attribute__((aligned(32)));

  t_nrLDPC_time_stats procTime = {0};
  t_nrLDPC_time_stats *p_procTime = &procTime;

  ////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////// nr_deinterleaving_ldpc ///////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////

  start_meas(rdata->p_ts_deinterleave);

  // code blocks after bit selection in rate matching for LDPC code (38.212 section 5.4.2.1)
  int16_t harq_e[E];

  nr_deinterleaving_ldpc(E, Qm, harq_e, ulsch_llr);

  stop_meas(rdata->p_ts_deinterleave);

  start_meas(rdata->p_ts_rate_unmatch);

  ////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////// nr_rate_matching_ldpc_rx //////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////

  if (nr_rate_matching_ldpc_rx(rdata->tbslbrm,
                               p_decoderParms->BG,
                               p_decoderParms->Z,
                               rdata->d,
                               harq_e,
                               rdata->C,
                               rv_index,
                               *rdata->d_to_be_cleared,
                               E,
                               rdata->F,
                               K - rdata->F - 2 * (p_decoderParms->Z))
      == -1) {
    stop_meas(rdata->p_ts_rate_unmatch);
    LOG_E(PHY, "nrLDPC_coding_segment_decoder.c: Problem in rate_matching\n");
    completed_task_ans(rdata->ans);
    return;
  }

  stop_meas(rdata->p_ts_rate_unmatch);

  *rdata->d_to_be_cleared = false;

  p_decoderParms->crc_type = crcType(rdata->C, A);
  p_decoderParms->Kprime = lenWithCrc(rdata->C, A);

  // set first 2*Z_c bits to zeros
  int16_t z[68 * 384 + 16] __attribute__((aligned(16)));

  start_meas(rdata->p_ts_ldpc_decode);

  memset(z, 0, 2 * rdata->Z * sizeof(*z));
  // set Filler bits
  memset(z + Kprime, 127, rdata->F * sizeof(*z));
  // Move coded bits before filler bits
  memcpy(z + 2 * rdata->Z, rdata->d, (Kprime - 2 * rdata->Z) * sizeof(*z));
  // skip filler bits
  memcpy(z + K, rdata->d + (K - 2 * rdata->Z), (Kc * rdata->Z - K) * sizeof(*z));

  // Saturate coded bits before decoding into 8-bit values
  simde__m128i *pv = (simde__m128i *)&z;
  int8_t l[68 * 384 + 16] __attribute__((aligned(16)));
  simde__m128i *pl = (simde__m128i *)&l;
  for (int i = 0, j = 0; j < ((Kc * rdata->Z) >> 4) + 1; i += 2, j++) {
    pl[j] = simde_mm_packs_epi16(pv[i], pv[i + 1]);
  }

  // ===== SPX debug oracle dumps (read-only; do NOT modify pipeline) =====
  if (spx_dump_en) {
    spx_prepare_dump_dirs();

    const int Zc = p_decoderParms->Z;
    const int BG = p_decoderParms->BG;
    const int out_len_i8 = (BG == 1) ? (68 * Zc) : (52 * Zc); // int8 elements
    const int out_len_i16 = out_len_i8;                       // z[] logical length / d[] logical length

    char fn_d[512];
    char fn_z[512];
    char fn_l[512];
    char fn_rm_exact[512];

    // UE oracle after nr_rate_matching_ldpc_rx(): d[]
    snprintf(fn_d, sizeof(fn_d),
             "%s/f%04u_s%02u_ldpc_ue_dunmatch_idx%06u_tb%02u_rv%01u_seg%02u_len%05d_i16.bin",
             spx_raw_dir,
             (unsigned)rdata->spx_frame,
             (unsigned)rdata->spx_slot,
             (unsigned)rdata->spx_idx,
             (unsigned)rdata->spx_tb,
             (unsigned)rdata->rv_index,
             (unsigned)rdata->r,
             out_len_i16);

    // UE oracle after z[] rebuild, before int8 pack
    snprintf(fn_z, sizeof(fn_z),
             "%s/f%04u_s%02u_ldpc_ue_z_idx%06u_tb%02u_rv%01u_seg%02u_len%05d_i16.bin",
             spx_raw_dir,
             (unsigned)rdata->spx_frame,
             (unsigned)rdata->spx_slot,
             (unsigned)rdata->spx_idx,
             (unsigned)rdata->spx_tb,
             (unsigned)rdata->rv_index,
             (unsigned)rdata->r,
             out_len_i16);

    // UE oracle exact int8 buffer fed to LDPCdecoder()
    snprintf(fn_l, sizeof(fn_l),
             "%s/f%04u_s%02u_ldpc_ue_l_idx%06u_tb%02u_rv%01u_seg%02u_outlen%05d_i8.bin",
             spx_raw_dir,
             (unsigned)rdata->spx_frame,
             (unsigned)rdata->spx_slot,
             (unsigned)rdata->spx_idx,
             (unsigned)rdata->spx_tb,
             (unsigned)rdata->rv_index,
             (unsigned)rdata->r,
             out_len_i8);

    // Keep original rm_exact filename for downstream ldpctest_spx pipeline compatibility
    snprintf(fn_rm_exact, sizeof(fn_rm_exact),
             "%s/f%04u_s%02u_ldpc_rm_exact_idx%06u_tb%02u_rv%01u_seg%02u_outlen%05d_i8.bin",
             spx_raw_dir,
             (unsigned)rdata->spx_frame,
             (unsigned)rdata->spx_slot,
             (unsigned)rdata->spx_idx,
             (unsigned)rdata->spx_tb,
             (unsigned)rdata->rv_index,
             (unsigned)rdata->r,
             out_len_i8);

    spx_dump_i16_file(fn_d, rdata->d, out_len_i16);
    spx_dump_i16_file(fn_z, z, out_len_i16);
    spx_dump_i8_file(fn_l, l, out_len_i8);
    spx_dump_i8_file(fn_rm_exact, l, out_len_i8);
  }
  // ===== SPX dump end =====

  // LDPC decode
  int decodeIterations = LDPCdecoder(p_decoderParms, l, llrProcBuf, p_procTime, rdata->abort_decode);

  if (decodeIterations < p_decoderParms->numMaxIter) {
    memcpy(rdata->c, llrProcBuf, K >> 3);
    *rdata->decodeSuccess = true;
  } else {
    memset(rdata->c, 0, K >> 3);
    *rdata->decodeSuccess = false;
  }

  stop_meas(rdata->p_ts_ldpc_decode);

  completed_task_ans(rdata->ans);
}

int nrLDPC_prepare_TB_decoding(nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters,
                               int pusch_id,
                               thread_info_tm_t *t_info)
{
  nrLDPC_TB_decoding_parameters_t *nrLDPC_TB_decoding_parameters = &nrLDPC_slot_decoding_parameters->TBs[pusch_id];

  *nrLDPC_TB_decoding_parameters->processedSegments = 0;
  t_nrLDPC_dec_params decParams = {.check_crc = check_crc};
  decParams.BG = nrLDPC_TB_decoding_parameters->BG;
  decParams.Z = nrLDPC_TB_decoding_parameters->Z;
  decParams.numMaxIter = nrLDPC_TB_decoding_parameters->max_ldpc_iterations;
  decParams.outMode = 0;

  // ===== SPX: use frame/slot from slot params =====
  const uint16_t spx_frame = (uint16_t)nrLDPC_slot_decoding_parameters->frame;
  const uint8_t  spx_slot  = (uint8_t)nrLDPC_slot_decoding_parameters->slot;
  const uint32_t spx_idx   = spx_make_idx(spx_frame, spx_slot);
  // ==============================================

  for (int r = 0; r < nrLDPC_TB_decoding_parameters->C; r++) {
    nrLDPC_decoding_parameters_t *rdata = &((nrLDPC_decoding_parameters_t *)t_info->buf)[t_info->len];
    DevAssert(t_info->len < t_info->cap);
    rdata->ans = t_info->ans;
    t_info->len += 1;

    decParams.R = nrLDPC_TB_decoding_parameters->segments[r].R;
    rdata->decoderParms = decParams;
    rdata->llr = nrLDPC_TB_decoding_parameters->segments[r].llr;
    rdata->Kc = decParams.BG == 2 ? 52 : 68;
    rdata->C = nrLDPC_TB_decoding_parameters->C;
    rdata->E = nrLDPC_TB_decoding_parameters->segments[r].E;
    rdata->A = nrLDPC_TB_decoding_parameters->A;
    rdata->Qm = nrLDPC_TB_decoding_parameters->Qm;
    rdata->K = nrLDPC_TB_decoding_parameters->K;
    rdata->Z = nrLDPC_TB_decoding_parameters->Z;
    rdata->F = nrLDPC_TB_decoding_parameters->F;
    rdata->rv_index = nrLDPC_TB_decoding_parameters->rv_index;
    rdata->tbslbrm = nrLDPC_TB_decoding_parameters->tbslbrm;
    rdata->abort_decode = nrLDPC_TB_decoding_parameters->abort_decode;
    rdata->d = nrLDPC_TB_decoding_parameters->segments[r].d;
    rdata->d_to_be_cleared = nrLDPC_TB_decoding_parameters->segments[r].d_to_be_cleared;
    rdata->c = nrLDPC_TB_decoding_parameters->segments[r].c;
    rdata->decodeSuccess = &nrLDPC_TB_decoding_parameters->segments[r].decodeSuccess;
    rdata->p_ts_deinterleave = &nrLDPC_TB_decoding_parameters->segments[r].ts_deinterleave;
    rdata->p_ts_rate_unmatch = &nrLDPC_TB_decoding_parameters->segments[r].ts_rate_unmatch;
    rdata->p_ts_ldpc_decode = &nrLDPC_TB_decoding_parameters->segments[r].ts_ldpc_decode;

    // ===== SPX: stable identity for dump naming =====
    rdata->spx_frame = spx_frame;
    rdata->spx_slot  = spx_slot;
    rdata->spx_idx   = spx_idx;
    rdata->spx_tb    = (uint8_t)pusch_id;
    rdata->r         = (uint8_t)r;
    // ==============================================

    task_t t = {.func = &nr_process_decode_segment, .args = rdata};
    pushTpool(nrLDPC_slot_decoding_parameters->threadPool, t);

    LOG_D(PHY, "Added a block to decode, in pipe: %d\n", r);
  }
  return nrLDPC_TB_decoding_parameters->C;
}

int32_t nrLDPC_coding_init(void)
{
  return 0;
}

int32_t nrLDPC_coding_shutdown(void)
{
  return 0;
}

int32_t nrLDPC_coding_decoder(nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters)
{
  int nbSegments = 0;
  for (int pusch_id = 0; pusch_id < nrLDPC_slot_decoding_parameters->nb_TBs; pusch_id++) {
    nrLDPC_TB_decoding_parameters_t *nrLDPC_TB_decoding_parameters = &nrLDPC_slot_decoding_parameters->TBs[pusch_id];
    nbSegments += nrLDPC_TB_decoding_parameters->C;
  }

  nrLDPC_decoding_parameters_t arr[nbSegments];

  task_ans_t ans;
  init_task_ans(&ans, nbSegments);
  thread_info_tm_t t_info = {.buf = (uint8_t *)arr, .len = 0, .cap = nbSegments, .ans = &ans};

  for (int pusch_id = 0; pusch_id < nrLDPC_slot_decoding_parameters->nb_TBs; pusch_id++) {
    (void)nrLDPC_prepare_TB_decoding(nrLDPC_slot_decoding_parameters, pusch_id, &t_info);
  }

  // Execute thread pool tasks
  join_task_ans(t_info.ans);

  for (int pusch_id = 0; pusch_id < nrLDPC_slot_decoding_parameters->nb_TBs; pusch_id++) {
    nrLDPC_TB_decoding_parameters_t *nrLDPC_TB_decoding_parameters = &nrLDPC_slot_decoding_parameters->TBs[pusch_id];
    for (int r = 0; r < nrLDPC_TB_decoding_parameters->C; r++) {
      if (nrLDPC_TB_decoding_parameters->segments[r].decodeSuccess) {
        *nrLDPC_TB_decoding_parameters->processedSegments =
            *nrLDPC_TB_decoding_parameters->processedSegments + 1;
      }
    }
  }

  return 0;
}
