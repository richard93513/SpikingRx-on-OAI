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
 */

/*! \file PHY/NR_UE_TRANSPORT/nr_dlsch_decoding.c
 */

#include "common/utils/LOG/vcd_signal_dumper.h"
#include "PHY/defs_nr_UE.h"
#include "SCHED_NR_UE/harq_nr.h"
#include "PHY/phy_extern_nr_ue.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "SCHED_NR_UE/defs.h"
#include "SIMULATION/TOOLS/sim.h"
#include "executables/nr-uesoftmodem.h"
#include "PHY/CODING/nrLDPC_extern.h"
#include "common/utils/nr/nr_common.h"
#include "openair1/PHY/TOOLS/phy_scope_interface.h"
#include "nfapi/open-nFAPI/nfapi/public_inc/nfapi_nr_interface.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

// ------------------------------------------------------------
// KPI
// ------------------------------------------------------------
static extended_kpi_ue kpiStructure = {0};

// ------------------------------------------------------------
// SPX dump control (UE side)
// ------------------------------------------------------------
static int spx_dump_en = 1;
static const char *spx_raw_dir = "/home/richard93513/SpikingRx-on-OAI/spx_records/raw";

static inline uint32_t spx_make_idx(int frame, int slot)
{
  return (uint32_t)frame * 100u + (uint32_t)slot;
}

static inline void spx_mkdirs(void)
{
  mkdir("/home/richard93513/SpikingRx-on-OAI", 0755);
  mkdir("/home/richard93513/SpikingRx-on-OAI/spx_records", 0755);
  mkdir(spx_raw_dir, 0755);
}

// ------------------------------------------------------------

extended_kpi_ue *getKPIUE(void)
{
  return &kpiStructure;
}

void nr_ue_dlsch_init(NR_UE_DLSCH_t *dlsch_list, int num_dlsch, uint8_t max_ldpc_iterations)
{
  for (int i = 0; i < num_dlsch; i++) {
    NR_UE_DLSCH_t *dlsch = dlsch_list + i;
    memset(dlsch, 0, sizeof(NR_UE_DLSCH_t));
    dlsch->max_ldpc_iterations = max_ldpc_iterations;
  }
}

void nr_dlsch_unscrambling(int16_t *llr, uint32_t size, uint8_t q, uint32_t Nid, uint32_t n_RNTI)
{
  nr_codeword_unscrambling(llr, size, q, Nid, n_RNTI);
}

/*! \brief Prepare necessary parameters for nrLDPC_coding_interface
 */
void nr_dlsch_decoding(PHY_VARS_NR_UE *phy_vars_ue,
                       const UE_nr_rxtx_proc_t *proc,
                       NR_UE_DLSCH_t *dlsch,
                       int16_t **dlsch_llr,
                       uint8_t **b,
                       int *G,
                       int nb_dlsch,
                       uint8_t *DLSCH_ids)
{
  // slot TB containers
  nrLDPC_TB_decoding_parameters_t TBs[nb_dlsch];
  memset(TBs, 0, sizeof(TBs));

  nrLDPC_slot_decoding_parameters_t slot_parameters = {
      .frame = proc->frame_rx,
      .slot = proc->nr_slot_rx,
      .nb_TBs = nb_dlsch,
      .threadPool = &get_nrUE_params()->Tpool,
      .TBs = TBs};

  int max_num_segments = 0;

  // ============================================================
  // Pre-decode: fill TB parameters + do segmentation if needed
  // ============================================================
  for (uint8_t pdsch_id = 0; pdsch_id < nb_dlsch; pdsch_id++) {
    uint8_t DLSCH_id = DLSCH_ids[pdsch_id];
    fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_config = &dlsch[DLSCH_id].dlsch_config;
    uint8_t dmrs_Type = dlsch_config->dmrsConfigType;
    int harq_pid = dlsch_config->harq_process_nbr;
    NR_DL_UE_HARQ_t *harq_process = &phy_vars_ue->dl_harq_processes[DLSCH_id][harq_pid];

    AssertFatal(dmrs_Type == 0 || dmrs_Type == 1, "Illegal dmrs_type %d\n", dmrs_Type);

    if (!harq_process) {
      LOG_E(PHY, "nr_dlsch_decoding.c: NULL harq_process pointer\n");
      return;
    }

    uint8_t nb_re_dmrs;
    if (dmrs_Type == NFAPI_NR_DMRS_TYPE1)
      nb_re_dmrs = 6 * dlsch_config->n_dmrs_cdm_groups;
    else
      nb_re_dmrs = 4 * dlsch_config->n_dmrs_cdm_groups;

    uint16_t dmrs_length = get_num_dmrs(dlsch_config->dlDmrsSymbPos);

    LOG_D(PHY, "Round %d RV idx %d\n", harq_process->DLround, dlsch_config->rv);

    nrLDPC_TB_decoding_parameters_t *TB_parameters = &TBs[pdsch_id];

    // unique pid: 2*harq_pid + DLSCH_id (DLSCH_id < 2)
    TB_parameters->harq_unique_pid = 2 * harq_pid + DLSCH_id;

    TB_parameters->G = G[DLSCH_id];
    TB_parameters->nb_rb = dlsch_config->number_rbs;
    TB_parameters->Qm = dlsch_config->qamModOrder;
    TB_parameters->mcs = dlsch_config->mcs;
    TB_parameters->nb_layers = dlsch[DLSCH_id].Nl;
    TB_parameters->BG = dlsch_config->ldpcBaseGraph;
    TB_parameters->A = dlsch_config->TBS;

    TB_parameters->processedSegments = &harq_process->processedSegments;

    float Coderate = (float)dlsch_config->targetCodeRate / 10240.0f;

    LOG_D(PHY,
          "%d.%d DLSCH %d Decoding, harq_pid %d TBS %d G %d nb_re_dmrs %d length dmrs %d mcs %d Nl %d nb_symb_sch %d nb_rb %d Qm %d "
          "Coderate %f\n",
          slot_parameters.frame,
          slot_parameters.slot,
          DLSCH_id,
          harq_pid,
          dlsch_config->TBS,
          TB_parameters->G,
          nb_re_dmrs,
          dmrs_length,
          TB_parameters->mcs,
          TB_parameters->nb_layers,
          dlsch_config->number_symbols,
          TB_parameters->nb_rb,
          TB_parameters->Qm,
          Coderate);

    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_DLSCH_SEGMENTATION, VCD_FUNCTION_IN);

    if (harq_process->first_rx == 1) {
      nr_segmentation(NULL,
                      NULL,
                      lenWithCrc(1, TB_parameters->A),
                      &TB_parameters->C,
                      &TB_parameters->K,
                      &TB_parameters->Z,
                      &TB_parameters->F,
                      TB_parameters->BG);

      harq_process->C = TB_parameters->C;
      harq_process->K = TB_parameters->K;
      harq_process->Z = TB_parameters->Z;
      harq_process->F = TB_parameters->F;

      if (harq_process->C > MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * TB_parameters->nb_layers) {
        LOG_E(PHY, "nr_segmentation.c: too many segments %d, A %d\n", harq_process->C, TB_parameters->A);
        return;
      }

      for (int i = 0; i < harq_process->C; i++)
        memset(harq_process->d[i], 0, 5 * 8448 * sizeof(int16_t));

    } else {
      TB_parameters->C = harq_process->C;
      TB_parameters->K = harq_process->K;
      TB_parameters->Z = harq_process->Z;
      TB_parameters->F = harq_process->F;
    }

    max_num_segments = max(max_num_segments, TB_parameters->C);

    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_DLSCH_SEGMENTATION, VCD_FUNCTION_OUT);

    TB_parameters->max_ldpc_iterations = dlsch[DLSCH_id].max_ldpc_iterations;
    TB_parameters->rv_index = dlsch_config->rv;
    TB_parameters->tbslbrm = dlsch_config->tbslbrm;
    TB_parameters->abort_decode = &harq_process->abort_decode;
    set_abort(&harq_process->abort_decode, false);
  }

  // ============================================================
  // Build segment parameters (E/R/llr pointers) for decoder
  // ============================================================
  nrLDPC_segment_decoding_parameters_t segments[nb_dlsch][max_num_segments];
  memset(segments, 0, sizeof(segments));

  bool d_to_be_cleared[nb_dlsch][max_num_segments];
  memset(d_to_be_cleared, 0, sizeof(d_to_be_cleared));

  for (uint8_t pdsch_id = 0; pdsch_id < nb_dlsch; pdsch_id++) {
    uint8_t DLSCH_id = DLSCH_ids[pdsch_id];
    fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_config = &dlsch[DLSCH_id].dlsch_config;
    int harq_pid = dlsch_config->harq_process_nbr;
    NR_DL_UE_HARQ_t *harq_process = &phy_vars_ue->dl_harq_processes[DLSCH_id][harq_pid];

    nrLDPC_TB_decoding_parameters_t *TB_parameters = &TBs[pdsch_id];
    TB_parameters->segments = segments[pdsch_id];

    uint32_t r_offset = 0;
    for (int r = 0; r < TB_parameters->C; r++) {
      d_to_be_cleared[pdsch_id][r] = (harq_process->first_rx == 1);

      nrLDPC_segment_decoding_parameters_t *segment_parameters = &TB_parameters->segments[r];

      segment_parameters->E = nr_get_E(TB_parameters->G,
                                       TB_parameters->C,
                                       TB_parameters->Qm,
                                       TB_parameters->nb_layers,
                                       r);

      segment_parameters->R = nr_get_R_ldpc_decoder(TB_parameters->rv_index,
                                                   segment_parameters->E,
                                                   TB_parameters->BG,
                                                   TB_parameters->Z,
                                                   &harq_process->llrLen,
                                                   harq_process->DLround);

      segment_parameters->llr = dlsch_llr[DLSCH_id] + r_offset;
      segment_parameters->d = harq_process->d[r];
      segment_parameters->d_to_be_cleared = &d_to_be_cleared[pdsch_id][r];
      segment_parameters->c = harq_process->c[r];
      segment_parameters->decodeSuccess = false;

      reset_meas(&segment_parameters->ts_deinterleave);
      reset_meas(&segment_parameters->ts_rate_unmatch);
      reset_meas(&segment_parameters->ts_ldpc_decode);

      r_offset += segment_parameters->E;
    }

    // ============================================================
    // SPX: Dump UE LDPC cfg json (once per f/s/dlsch/harq/round)
    // ============================================================
    if (spx_dump_en) {
      static int last_f = -1, last_s = -1, last_d = -1, last_h = -1, last_round = -1;
      const int cur_round = (int)harq_process->DLround;

      if (!(last_f == proc->frame_rx && last_s == proc->nr_slot_rx &&
            last_d == (int)DLSCH_id && last_h == (int)harq_pid && last_round == cur_round)) {

        last_f = proc->frame_rx;
        last_s = proc->nr_slot_rx;
        last_d = (int)DLSCH_id;
        last_h = (int)harq_pid;
        last_round = cur_round;

        spx_mkdirs();
        const uint32_t idx = spx_make_idx(proc->frame_rx, proc->nr_slot_rx);

        char cfg_path[512];
        snprintf(cfg_path, sizeof(cfg_path),
                 "%s/f%04d_s%02d_ldpc_idx%06u_rnti%04x_dlsch%02d_harq%02d_round%02d_rv%01d.json",
                 spx_raw_dir,
                 proc->frame_rx, proc->nr_slot_rx, idx,
                 (unsigned)(dlsch[DLSCH_id].rnti & 0xffff),
                 (int)DLSCH_id, (int)harq_pid, cur_round, (int)TB_parameters->rv_index);

        FILE *fp = fopen(cfg_path, "w");
        if (!fp) {
          LOG_E(PHY, "[SPX][ERR] cannot open %s for UE LDPC cfg dump (errno=%d)\n", cfg_path, errno);
        } else {
          fprintf(fp, "{\n");
          fprintf(fp, "  \"side\": \"UE\",\n");
          fprintf(fp, "  \"frame\": %d,\n", proc->frame_rx);
          fprintf(fp, "  \"slot\": %d,\n", proc->nr_slot_rx);
          fprintf(fp, "  \"rnti\": %u,\n", (unsigned int)dlsch[DLSCH_id].rnti);
          fprintf(fp, "  \"dlsch_id\": %d,\n", (int)DLSCH_id);
          fprintf(fp, "  \"harq_pid\": %d,\n", (int)harq_pid);
          fprintf(fp, "  \"round\": %d,\n", cur_round);

          fprintf(fp, "  \"BG\": %d,\n", TB_parameters->BG);
          fprintf(fp, "  \"Zc\": %d,\n", TB_parameters->Z);
          fprintf(fp, "  \"A\": %d,\n", TB_parameters->A);
          fprintf(fp, "  \"C\": %d,\n", TB_parameters->C);
          fprintf(fp, "  \"K\": %d,\n", TB_parameters->K);
          fprintf(fp, "  \"F\": %d,\n", TB_parameters->F);

          fprintf(fp, "  \"G\": %d,\n", TB_parameters->G);
          fprintf(fp, "  \"Qm\": %d,\n", TB_parameters->Qm);
          fprintf(fp, "  \"nb_layers\": %d,\n", TB_parameters->nb_layers);
          fprintf(fp, "  \"rv_index\": %d,\n", TB_parameters->rv_index);
          fprintf(fp, "  \"tbslbrm\": %d,\n", TB_parameters->tbslbrm);
          fprintf(fp, "  \"mcs\": %d,\n", TB_parameters->mcs);
          fprintf(fp, "  \"llrLen\": %d,\n", harq_process->llrLen);

          fprintf(fp, "  \"E_list\": [");
          for (int r = 0; r < TB_parameters->C; r++)
            fprintf(fp, "%d%s", TB_parameters->segments[r].E, (r == TB_parameters->C - 1) ? "" : ", ");
          fprintf(fp, "],\n");

          fprintf(fp, "  \"R_list\": [");
          for (int r = 0; r < TB_parameters->C; r++)
            fprintf(fp, "%d%s", TB_parameters->segments[r].R, (r == TB_parameters->C - 1) ? "" : ", ");
          fprintf(fp, "]\n");

          fprintf(fp, "}\n");
          fclose(fp);

          LOG_I(PHY, "[SPX] Dumped UE LDPC cfg: f=%d s=%d idx=%06u dlsch=%d harq=%d round=%d -> %s\n",
                proc->frame_rx, proc->nr_slot_rx, idx, (unsigned int)DLSCH_id, (unsigned int)harq_pid, cur_round, cfg_path);
        }
      }
    }
    // ============================
  }

  // ============================================================
  // Call decoder
  // ============================================================
  int ret_decoder = phy_vars_ue->nrLDPC_coding_interface.nrLDPC_coding_decoder(&slot_parameters);
  if (ret_decoder != 0) {
    LOG_E(PHY, "nrLDPC_coding_decoder returned an error: %d\n", ret_decoder);
    return;
  }

  // ============================================================
  // Post decode: reconstruct TB, dump ue_c and ue_tb
  // ============================================================
  for (uint8_t pdsch_id = 0; pdsch_id < nb_dlsch; pdsch_id++) {
    uint8_t DLSCH_id = DLSCH_ids[pdsch_id];
    fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_config = &dlsch[DLSCH_id].dlsch_config;
    int harq_pid = dlsch_config->harq_process_nbr;
    NR_DL_UE_HARQ_t *harq_process = &phy_vars_ue->dl_harq_processes[DLSCH_id][harq_pid];

    nrLDPC_TB_decoding_parameters_t *TB_parameters = &TBs[pdsch_id];

    uint32_t offset = 0;
    for (int r = 0; r < TB_parameters->C; r++) {
      nrLDPC_segment_decoding_parameters_t *segment_parameters = &TB_parameters->segments[r];

      const int Kr_bytes = (harq_process->K >> 3) - (harq_process->F >> 3) - ((harq_process->C > 1) ? 3 : 0);

      if (segment_parameters->decodeSuccess) {
        // UE reconstruct TB bytes from c[r] into b+offset
        memcpy(b[DLSCH_id] + offset, harq_process->c[r], Kr_bytes);

        // --- SPX: dump UE decoded CB payload bytes (ue_c) ---
        if (spx_dump_en) {
          spx_mkdirs();
          const uint32_t idx = spx_make_idx(proc->frame_rx, proc->nr_slot_rx);

          char fn[512];
          snprintf(fn, sizeof(fn),
                   "%s/f%04d_s%02d_ue_c_idx%06u_rnti%04x_dlsch%02d_harq%02d_round%02d_rv%01d_seg%02d.bin",
                   spx_raw_dir,
                   proc->frame_rx, proc->nr_slot_rx,
                   idx,
                   (unsigned)(dlsch[DLSCH_id].rnti & 0xffff),
                   (int)DLSCH_id,
                   (int)harq_pid,
                   (int)harq_process->DLround,
                   (int)TB_parameters->rv_index,
                   r);

          FILE *fp = fopen(fn, "wb");
          if (!fp) {
            LOG_E(PHY, "[SPX][ERR] cannot open %s for UE c dump (errno=%d)\n", fn, errno);
          } else {
            fwrite((void *)(b[DLSCH_id] + offset), 1, (size_t)Kr_bytes, fp);
            fclose(fp);
            LOG_I(PHY, "[SPX] Dumped UE decoded CB payload: f=%d s=%d idx=%06u seg=%d Kr_bytes=%d -> %s\n",
                  proc->frame_rx, proc->nr_slot_rx, idx, r, Kr_bytes, fn);
          }
        }

      } else {
        LOG_D(PHY, "frame=%d, slot=%d, first_rx=%d, rv_index=%d\n",
              proc->frame_rx, proc->nr_slot_rx, harq_process->first_rx, dlsch_config->rv);
        LOG_D(PHY, "downlink segment error %d/%d\n", r, harq_process->C);
        LOG_D(PHY, "DLSCH %d in error\n", DLSCH_id);
      }

      offset += Kr_bytes;

      merge_meas(&phy_vars_ue->phy_cpu_stats.cpu_time_stats[DLSCH_DEINTERLEAVING_STATS], &segment_parameters->ts_deinterleave);
      merge_meas(&phy_vars_ue->phy_cpu_stats.cpu_time_stats[DLSCH_RATE_UNMATCHING_STATS], &segment_parameters->ts_rate_unmatch);
      merge_meas(&phy_vars_ue->phy_cpu_stats.cpu_time_stats[DLSCH_LDPC_DECODING_STATS], &segment_parameters->ts_ldpc_decode);
    }

    kpiStructure.nb_total++;
    kpiStructure.blockSize = dlsch_config->TBS;
    kpiStructure.dl_mcs = dlsch_config->mcs;
    kpiStructure.nofRBs = dlsch_config->number_rbs;

    harq_process->decodeResult = harq_process->processedSegments == harq_process->C;

    if (harq_process->decodeResult && harq_process->C > 1) {
      int A = dlsch[DLSCH_id].dlsch_config.TBS;
      if (!check_crc(b[DLSCH_id], lenWithCrc(1, A), crcType(1, A))) {
        LOG_E(PHY,
              " Frame %d.%d LDPC global CRC fails, but individual LDPC CRC succeeded. %d segs\n",
              proc->frame_rx,
              proc->nr_slot_rx,
              harq_process->C);
        harq_process->decodeResult = false;
      }
    }

    if (harq_process->decodeResult) {
      // --- SPX: dump UE reconstructed TB bytes (ue_tb) ---
      if (spx_dump_en) {
        spx_mkdirs();
        const uint32_t idx = spx_make_idx(proc->frame_rx, proc->nr_slot_rx);
        const int tbs_bytes = dlsch[DLSCH_id].dlsch_config.TBS / 8;

        char fn[512];
        snprintf(fn, sizeof(fn),
                 "%s/f%04d_s%02d_ue_tb_idx%06u_rnti%04x_dlsch%02d_harq%02d_round%02d_rv%01d.bin",
                 spx_raw_dir,
                 proc->frame_rx, proc->nr_slot_rx,
                 idx,
                 (unsigned)(dlsch[DLSCH_id].rnti & 0xffff),
                 (int)DLSCH_id,
                 (int)harq_pid,
                 (int)harq_process->DLround,
                 (int)TB_parameters->rv_index);

        FILE *fp = fopen(fn, "wb");
        if (!fp) {
          LOG_E(PHY, "[SPX][ERR] cannot open %s for UE TB dump (errno=%d)\n", fn, errno);
        } else {
          fwrite(b[DLSCH_id], 1, (size_t)tbs_bytes, fp);
          fclose(fp);
          LOG_I(PHY, "[SPX] Dumped UE TB bytes: f=%d s=%d idx=%06u tbs_bytes=%d -> %s\n",
                proc->frame_rx, proc->nr_slot_rx, idx, tbs_bytes, fn);
        }
      }

      LOG_D(PHY, "DLSCH received ok \n");
      harq_process->status = SCH_IDLE;
      dlsch[DLSCH_id].last_iteration_cnt = dlsch[DLSCH_id].max_ldpc_iterations - 1;
    } else {
      LOG_D(PHY, "DLSCH received nok \n");
      kpiStructure.nb_nack++;
      dlsch[DLSCH_id].last_iteration_cnt = dlsch[DLSCH_id].max_ldpc_iterations;
      UEdumpScopeData(phy_vars_ue, proc->nr_slot_rx, proc->frame_rx, "DLSCH_NACK");
    }

    uint8_t dmrs_Type = dlsch_config->dmrsConfigType;
    uint8_t nb_re_dmrs;
    if (dmrs_Type == NFAPI_NR_DMRS_TYPE1)
      nb_re_dmrs = 6 * dlsch_config->n_dmrs_cdm_groups;
    else
      nb_re_dmrs = 4 * dlsch_config->n_dmrs_cdm_groups;

    uint16_t dmrs_length = get_num_dmrs(dlsch_config->dlDmrsSymbPos);
    float Coderate = (float)dlsch_config->targetCodeRate / 10240.0f;

    LOG_D(PHY,
          "%d.%d DLSCH Decoded, harq_pid %d, round %d, result: %d TBS %d (%d) G %d nb_re_dmrs %d length dmrs %d mcs %d Nl %d "
          "nb_symb_sch %d nb_rb %d Qm %d Coderate %f\n",
          proc->frame_rx,
          proc->nr_slot_rx,
          harq_pid,
          harq_process->DLround,
          harq_process->decodeResult,
          dlsch[DLSCH_id].dlsch_config.TBS,
          dlsch[DLSCH_id].dlsch_config.TBS / 8,
          G[DLSCH_id],
          nb_re_dmrs,
          dmrs_length,
          dlsch_config->mcs,
          dlsch[DLSCH_id].Nl,
          dlsch_config->number_symbols,
          dlsch_config->number_rbs,
          dlsch_config->qamModOrder,
          Coderate);
  }
}
