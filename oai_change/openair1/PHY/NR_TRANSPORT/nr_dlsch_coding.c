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

/*! \file PHY/NR_TRANSPORT/nr_dlsch_coding_slot.c
 * \brief Top-level routines for implementing LDPC-coded (DLSCH) transport channels from 38-212, 15.2
 *
 * NOTE (SPX):
 *   - Added dumps for TX TB bytes and gNB LDPC cfg JSON into:
 *       /home/richard93513/SpikingRx-on-OAI/spx_records/raw
 *   - Deterministic idx: idx = frame*100 + slot
 *   - TB CRC field name differs across OAI versions. To avoid compile breaks,
 *     this file defaults tbcrc=0 in dumps unless you define SPX_TBCRC_EXPR.
 */

#include "PHY/defs_gNB.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"
#include "PHY/CODING/nrLDPC_extern.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "SCHED_NR/sched_nr.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "common/utils/LOG/log.h"
#include "common/utils/nr/nr_common.h"
#include <syscall.h>
#include <openair2/UTIL/OPT/opt.h>

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

// ======================================================
// ======== SPX helpers =================================
// ======================================================

static inline uint32_t spx_make_idx(int frame, int slot)
{
  return (uint32_t)frame * 100u + (uint32_t)slot;
}

// Avoid dumping duplicates for same (frame,slot,rnti,pdu,rv,tbcrc)
static int spx_last_frame = -1;
static int spx_last_slot  = -1;
static int spx_last_rnti  = -1;
static int spx_last_pdu   = -1;
static int spx_last_rv    = -1;
static uint32_t spx_last_tbcrc = 0xffffffffu;

static inline void spx_mkdir_p(const char *path, mode_t mode)
{
  if (mkdir(path, mode) != 0 && errno != EEXIST) {
    LOG_W(PHY, "[SPX] mkdir(%s) failed: %s\n", path, strerror(errno));
  }
}

static inline void spx_prepare_rawdir(void)
{
  spx_mkdir_p("/home/richard93513/SpikingRx-on-OAI", 0755);
  spx_mkdir_p("/home/richard93513/SpikingRx-on-OAI/spx_records", 0755);
  spx_mkdir_p("/home/richard93513/SpikingRx-on-OAI/spx_records/raw", 0755);
}

/*
 * TB CRC field differs across OAI versions (eg: dlTbCrc, tb_crc, tbCrc, ...).
 * To keep this file "drop-in compile safe", default is 0.
 *
 * If you WANT tbcrc from rel15, build with for example:
 *   -DSPX_TBCRC_EXPR="(uint32_t)(rel15->dlTbCrc)"
 * or:
 *   -DSPX_TBCRC_EXPR="(uint32_t)(rel15->tb_crc)"
 */
#ifndef SPX_TBCRC_EXPR
#define SPX_TBCRC_EXPR (0u)
#endif

static inline uint32_t spx_get_tbcrc(const nfapi_nr_dl_tti_pdsch_pdu_rel15_t *rel15)
{
  (void)rel15;
  return (uint32_t)(SPX_TBCRC_EXPR);
}

// ======================================================
// === free_gNB_dlsch ===================================
// ======================================================
void free_gNB_dlsch(NR_gNB_DLSCH_t *dlsch, uint16_t N_RB, const NR_DL_FRAME_PARMS *frame_parms)
{
  int max_layers = (frame_parms->nb_antennas_tx < NR_MAX_NB_LAYERS) ?
                    frame_parms->nb_antennas_tx : NR_MAX_NB_LAYERS;
  uint16_t a_segments = MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * max_layers;

  if (N_RB != 273) {
    a_segments = (a_segments * N_RB) / 273 + 1;
  }

  if (dlsch->b) {
    free16(dlsch->b, a_segments * 1056);
    dlsch->b = NULL;
  }
  if (dlsch->f) {
    free16(dlsch->f,
           N_RB *
           NR_SYMBOLS_PER_SLOT *
           NR_NB_SC_PER_RB *
           8 *
           NR_MAX_NB_LAYERS);
    dlsch->f = NULL;
  }
  for (int r = 0; r < a_segments; r++) {
    free(dlsch->c[r]);
    dlsch->c[r] = NULL;
  }
  free(dlsch->c);
}

// ======================================================
// === new_gNB_dlsch ====================================
// ======================================================
NR_gNB_DLSCH_t new_gNB_dlsch(NR_DL_FRAME_PARMS *frame_parms, uint16_t N_RB)
{
  int max_layers = (frame_parms->nb_antennas_tx < NR_MAX_NB_LAYERS) ?
                    frame_parms->nb_antennas_tx : NR_MAX_NB_LAYERS;

  uint16_t a_segments =
      MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * max_layers;

  if (N_RB != 273) {
    a_segments = (a_segments * N_RB) / 273 + 1;
  }

  uint32_t dlsch_bytes = a_segments * 1056;
  NR_gNB_DLSCH_t dlsch = (NR_gNB_DLSCH_t){0};

  dlsch.b = malloc16(dlsch_bytes);
  AssertFatal(dlsch.b, "cannot allocate dlsch.b\n");
  bzero(dlsch.b, dlsch_bytes);

  dlsch.c = (uint8_t **)malloc16(a_segments * sizeof(uint8_t *));
  for (int r = 0; r < a_segments; r++) {
    dlsch.c[r] = malloc16(8448);
    AssertFatal(dlsch.c[r], "cannot allocate dlsch.c[%d]\n", r);
    bzero(dlsch.c[r], 8448);
  }

  dlsch.f = malloc16(
      N_RB *
      NR_SYMBOLS_PER_SLOT *
      NR_NB_SC_PER_RB *
      8 *
      NR_MAX_NB_LAYERS);
  AssertFatal(dlsch.f, "cannot allocate dlsch->f\n");
  bzero(dlsch.f,
        N_RB *
        NR_SYMBOLS_PER_SLOT *
        NR_NB_SC_PER_RB *
        8 *
        NR_MAX_NB_LAYERS);

  return dlsch;
}

// ======================================================
// === nr_dlsch_encoding() ==============================
// ======================================================
int nr_dlsch_encoding(PHY_VARS_gNB *gNB,
                      int n_dlsch,
                      NR_gNB_DLSCH_t *dlsch_array,
                      int frame,
                      uint8_t slot,
                      NR_DL_FRAME_PARMS *frame_parms,
                      unsigned char *output,
                      time_stats_t *tinput,
                      time_stats_t *tprep,
                      time_stats_t *tparity,
                      time_stats_t *toutput,
                      time_stats_t *dlsch_rate_matching_stats,
                      time_stats_t *dlsch_interleaving_stats,
                      time_stats_t *dlsch_segmentation_stats)
{
  (void)frame_parms;

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(
      VCD_SIGNAL_DUMPER_FUNCTIONS_gNB_DLSCH_ENCODING,
      VCD_FUNCTION_IN);

  nrLDPC_TB_encoding_parameters_t TBs[n_dlsch];
  memset(TBs, 0, sizeof(TBs));

  int num_segments = 0;

  // ------------------------------------------------------------------
  // TB loop: CRC + segmentation
  // ------------------------------------------------------------------
  for (int i = 0; i < n_dlsch; i++) {
    NR_gNB_DLSCH_t *dlsch = &dlsch_array[i];

    unsigned int crc = 1;
    const nfapi_nr_dl_tti_pdsch_pdu_rel15_t *rel15 =
        &dlsch->pdsch_pdu->pdsch_pdu_rel15;

    uint32_t A = rel15->TBSize[0] << 3;   // TB size in bits
    unsigned char *a = dlsch->pdu;        // TB bytes pointer (random bits)

    // ==========================================================
    // SPX: Dump TX TB bytes (after random, before CRC/LDPC)
    // ==========================================================
    {
      const char *raw_dir = "/home/richard93513/SpikingRx-on-OAI/spx_records/raw";
      spx_prepare_rawdir();

      const int rv_index   = (int)rel15->rvIndex[0];
      const int pdu_index  = (int)rel15->pduIndex;
      const int rnti       = (int)rel15->rnti;

      const uint32_t tbcrc = spx_get_tbcrc(rel15);
      const uint32_t this_idx = spx_make_idx(frame, slot);

      if (spx_last_frame == frame && spx_last_slot == slot &&
          spx_last_rnti == rnti && spx_last_pdu == pdu_index &&
          spx_last_rv == rv_index && spx_last_tbcrc == tbcrc) {
        goto spx_txbits_dump_end;
      }

      spx_last_frame = frame;
      spx_last_slot  = slot;
      spx_last_rnti  = rnti;
      spx_last_pdu   = pdu_index;
      spx_last_rv    = rv_index;
      spx_last_tbcrc = tbcrc;

      const unsigned rnti_u = (unsigned)(rnti & 0xffff);

      char fname[512];
      snprintf(fname, sizeof(fname),
           "%s/f%04d_s%02d_txbits_idx%06u_rnti%04x_pdu%03d_rv%01d_tbcrc%08x.bin",
           raw_dir, frame, slot, this_idx, rnti_u, pdu_index, rv_index, tbcrc);

      FILE *fp2 = fopen(fname, "wb");
      if (fp2) {
        const size_t tb_bytes = (size_t)rel15->TBSize[0];
        if (tb_bytes == 0) {
          LOG_W(PHY, "[SPX] TBSize[0] is 0, skip TX TB dump\n");
        } else {
          const size_t nwrite = fwrite(a, 1, tb_bytes, fp2);
          if (nwrite != tb_bytes) {
            LOG_W(PHY,"[SPX] TX TB dump short write: expect=%zuB wrote=%zuB (frame=%d slot=%d rnti=%04x pdu=%03d rv=%d tbcrc=0x%08x)\n",
      tb_bytes, nwrite, frame, slot, rnti_u, pdu_index, rv_index, tbcrc);
          }
        }
        fclose(fp2);

        LOG_I(PHY,"[SPX] Dumped TX TB bytes: frame=%d slot=%d rnti=%04x pdu=%03d rv=%d tbcrc=0x%08x A=%u bits bytes=%zu -> %s (idx=%06u)\n",
      frame, slot, rnti_u, pdu_index, rv_index, tbcrc, A, (size_t)rel15->TBSize[0], fname, this_idx);
      } else {
        LOG_W(PHY, "[SPX] Failed to open TX TB dump file: %s (%s)\n",
              fname, strerror(errno));
      }

spx_txbits_dump_end:
      ;
    }
    // ---------------- END SPX dump TX TB bytes ------------------

    if (rel15->rnti != SI_RNTI) {
      ws_trace_t tmp = {
        .nr = true,
        .direction = DIRECTION_DOWNLINK,
        .pdu_buffer = a,
        .pdu_buffer_size = rel15->TBSize[0],
        .ueid = 0,
        .rntiType = WS_C_RNTI,
        .rnti = rel15->rnti,
        .sysFrame = frame,
        .subframe = slot,
        .harq_pid = 0, // difficult to find the harq pid here
        .oob_event = 0,
        .oob_event_value = 0
      };
      trace_pdu(&tmp);
    }

    NR_gNB_PHY_STATS_t *phy_stats = NULL;
    if (rel15->rnti != 0xFFFF)
      phy_stats = get_phy_stats(gNB, rel15->rnti);

    if (phy_stats) {
      phy_stats->frame = frame;
      phy_stats->dlsch_stats.total_bytes_tx += rel15->TBSize[0];
      phy_stats->dlsch_stats.current_RI = rel15->nrOfLayers;
      phy_stats->dlsch_stats.current_Qm = rel15->qamModOrder[0];
    }

    int max_bytes =
        MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * rel15->nrOfLayers * 1056;
    int B;

    if (A > NR_MAX_PDSCH_TBS) {
      crc = crc24a(a, A) >> 8;
      a[A >> 3]         = ((uint8_t *)&crc)[2];
      a[1 + (A >> 3)]   = ((uint8_t *)&crc)[1];
      a[2 + (A >> 3)]   = ((uint8_t *)&crc)[0];
      B = A + 24;

      AssertFatal((A / 8) + 4 <= max_bytes,
                  "A %d is too big (A/8+4 = %d > %d)\n",
                  A, (A / 8) + 4, max_bytes);

      memcpy(dlsch->b, a, (A / 8) + 4);
    } else {
      crc = crc16(a, A) >> 16;
      a[A >> 3]       = ((uint8_t *)&crc)[1];
      a[1 + (A >> 3)] = ((uint8_t *)&crc)[0];
      B = A + 16;

      AssertFatal((A / 8) + 3 <= max_bytes,
                  "A %d is too big (A/8+3 = %d > %d)\n",
                  A, (A / 8) + 3, max_bytes);

      memcpy(dlsch->b, a, (A / 8) + 3);
    }

    nrLDPC_TB_encoding_parameters_t *TB_parameters = &TBs[i];

    TB_parameters->harq_unique_pid = i;
    TB_parameters->BG              = rel15->maintenance_parms_v3.ldpcBaseGraph;
    TB_parameters->A               = A;

    start_meas(dlsch_segmentation_stats);
    TB_parameters->Kb = nr_segmentation(dlsch->b,
                                        dlsch->c,
                                        B,
                                        &TB_parameters->C,
                                        &TB_parameters->K,
                                        &TB_parameters->Z,
                                        &TB_parameters->F,
                                        TB_parameters->BG);
    stop_meas(dlsch_segmentation_stats);

    if (TB_parameters->C >
        MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * rel15->nrOfLayers) {
      LOG_E(PHY,
            "nr_segmentation.c: too many segments %d, B %d\n",
            TB_parameters->C, B);
      return -1;
    }
    num_segments += TB_parameters->C;
  }

  // ------------------------------------------------------------------
  // Allocate segment parameters for all TBs
  // ------------------------------------------------------------------
  nrLDPC_segment_encoding_parameters_t segments[num_segments];
  memset(segments, 0, sizeof(segments));
  size_t segments_offset = 0;
  size_t dlsch_offset    = 0;

  // ------------------------------------------------------------------
  // Fill TB parameters + SPX dump LDPC cfg + attach segment params
  // ------------------------------------------------------------------
  for (int i = 0; i < n_dlsch; i++) {
    NR_gNB_DLSCH_t *dlsch = &dlsch_array[i];
    const nfapi_nr_dl_tti_pdsch_pdu_rel15_t *rel15 =
        &dlsch->pdsch_pdu->pdsch_pdu_rel15;

    nrLDPC_TB_encoding_parameters_t *TB_parameters = &TBs[i];

    TB_parameters->nb_rb     = rel15->rbSize;
    TB_parameters->Qm        = rel15->qamModOrder[0];
    TB_parameters->mcs       = rel15->mcsIndex[0];
    TB_parameters->nb_layers = rel15->nrOfLayers;
    TB_parameters->rv_index  = rel15->rvIndex[0];

    int nb_re_dmrs =
      (rel15->dmrsConfigType == NFAPI_NR_DMRS_TYPE1)
        ? (6 * rel15->numDmrsCdmGrpsNoData)
        : (4 * rel15->numDmrsCdmGrpsNoData);

    TB_parameters->G = nr_get_G(rel15->rbSize,
                               rel15->NrOfSymbols,
                               nb_re_dmrs,
                               get_num_dmrs(rel15->dlDmrsSymbPos),
                               dlsch->unav_res,
                               rel15->qamModOrder[0],
                               rel15->nrOfLayers);

    TB_parameters->tbslbrm =
        rel15->maintenance_parms_v3.tbSizeLbrmBytes;

// =====================================================
// SPX: Dump gNB LDPC config JSON (per TB)
// =====================================================
{
  spx_prepare_rawdir();
  const char *raw_dir = "/home/richard93513/SpikingRx-on-OAI/spx_records/raw";

  const int rv_index  = (int)rel15->rvIndex[0];
  const int pdu_index = (int)rel15->pduIndex;
  const int rnti      = (int)rel15->rnti;
  const unsigned rnti_u = (unsigned)(rnti & 0xffff);  // <-- 統一 16-bit RNTI
  const uint32_t tbcrc = spx_get_tbcrc(rel15);

  const uint32_t this_idx = spx_make_idx(frame, slot);

  char json_name[512];
  snprintf(json_name, sizeof(json_name),
           "%s/f%04d_s%02d_ldpc_idx%06u_rnti%04x_pdu%03d_rv%01d_tbcrc%08x.json",
           raw_dir, frame, slot, this_idx,
           rnti_u, pdu_index, rv_index, tbcrc);

  FILE *fj = fopen(json_name, "w");
  if (fj) {
    fprintf(fj,
            "{\n"
            "  \"side\": \"gNB\",\n"
            "  \"frame\": %d,\n"
            "  \"slot\": %d,\n"
            "  \"rnti\": %u,\n"
            "  \"pdu_index\": %d,\n"
            "  \"rv_index\": %d,\n"
            "  \"tbcrc\": %u,\n"
            "  \"BG\": %d,\n"
            "  \"Zc\": %d,\n"
            "  \"A\": %d,\n"
            "  \"C\": %d,\n"
            "  \"K\": %d,\n"
            "  \"F\": %d,\n"
            "  \"G\": %d,\n"
            "  \"Qm\": %d,\n"
            "  \"tbslbrm\": %d,\n"
            "  \"mcs\": %d,\n"
            "  \"nb_rb\": %d\n"
            "}\n",
            frame, slot,
            rnti_u, pdu_index, rv_index, tbcrc,
            TB_parameters->BG, TB_parameters->Z, TB_parameters->A, TB_parameters->C,
            TB_parameters->K, TB_parameters->F, TB_parameters->G,
            TB_parameters->Qm, TB_parameters->tbslbrm,
            TB_parameters->mcs, TB_parameters->nb_rb);
    fclose(fj);

    LOG_I(PHY,
          "[SPX] Dumped gNB LDPC cfg: frame=%d slot=%d rnti=%04x pdu=%03d rv=%d tbcrc=%u -> %s (idx=%06u)\n",
          frame, slot, rnti_u, pdu_index, rv_index, tbcrc, json_name, this_idx);
  } else {
    LOG_W(PHY, "[SPX] Failed to open gNB LDPC cfg file: %s (%s)\n",
          json_name, strerror(errno));
  }
}
// ---------------- END SPX LDPC cfg dump ----------------------

    TB_parameters->output   = &output[dlsch_offset >> 3];
    TB_parameters->segments = &segments[segments_offset];

    for (int r = 0; r < TB_parameters->C; r++) {
      nrLDPC_segment_encoding_parameters_t *segment_parameters =
          &TB_parameters->segments[r];
      segment_parameters->c = dlsch->c[r];
      segment_parameters->E = nr_get_E(TB_parameters->G,
                                       TB_parameters->C,
                                       TB_parameters->Qm,
                                       rel15->nrOfLayers,
                                       r);

      reset_meas(&segment_parameters->ts_interleave);
      reset_meas(&segment_parameters->ts_rate_match);
      reset_meas(&segment_parameters->ts_ldpc_encode);
    }

    segments_offset += TB_parameters->C;

    const size_t dlsch_size =
      (size_t)rel15->rbSize *
      (size_t)NR_SYMBOLS_PER_SLOT *
      (size_t)NR_NB_SC_PER_RB *
      (size_t)rel15->qamModOrder[0] *
      (size_t)rel15->nrOfLayers;

    dlsch_offset += ceil_mod(dlsch_size, 8 * 64);
  }

  // ------------------------------------------------------------------
  // Call LDPC encoder
  // ------------------------------------------------------------------
  nrLDPC_slot_encoding_parameters_t slot_parameters = {
    .frame      = frame,
    .slot       = slot,
    .nb_TBs     = n_dlsch,
    .threadPool = &gNB->threadPool,
    .tinput     = tinput,
    .tprep      = tprep,
    .tparity    = tparity,
    .toutput    = toutput,
    .TBs        = TBs
  };

  gNB->nrLDPC_coding_interface.nrLDPC_coding_encoder(&slot_parameters);

  // ------------------------------------------------------------------
  // Merge measurements
  // ------------------------------------------------------------------
  for (int i = 0; i < n_dlsch; i++) {
    nrLDPC_TB_encoding_parameters_t *TB_parameters = &TBs[i];
    for (int r = 0; r < TB_parameters->C; r++) {
      nrLDPC_segment_encoding_parameters_t *segment_parameters =
          &TB_parameters->segments[r];
      merge_meas(dlsch_interleaving_stats,
                 &segment_parameters->ts_interleave);
      merge_meas(dlsch_rate_matching_stats,
                 &segment_parameters->ts_rate_match);
    }
  }

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(
      VCD_SIGNAL_DUMPER_FUNCTIONS_gNB_DLSCH_ENCODING,
      VCD_FUNCTION_OUT);

  return 0;
}
