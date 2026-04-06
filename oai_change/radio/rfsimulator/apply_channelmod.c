/*
* Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
* contributor license agreements.  See the NOTICE file distributed with
* this work for additional information regarding copyright ownership.
* The OpenAirInterface Software Alliance licenses this file to You under
* the OAI Public License, Version 1.1  (the "License"); you may not use this file
* except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.openairinterface.org/?page_id=698
*
* Author and copyright: Laurent Thomas, open-cells.com
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

#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <common/utils/LOG/log.h>
#include <openair1/SIMULATION/TOOLS/sim.h>
#include "openair2/LAYER2/NR_MAC_gNB/mac_config.h"
#include "rfsimulator.h"

#define MAX_SNR_SAMPLES          50000
#define TARGET_SNR_WINDOWS       8
#define MIN_VALID_SNR_SAMPLES    2048

static int cmp_double_desc(const void *a, const void *b)
{
  const double da = *(const double *)a;
  const double db = *(const double *)b;
  if (db > da) return 1;
  if (db < da) return -1;
  return 0;
}

static void spx_dump_snr_summary(double noise_power_db, double snr_top10_db, int used_windows)
{
  const char *home = getenv("HOME");
  char path[1024];

  if (home && home[0] != '\0')
    snprintf(path, sizeof(path), "%s/SpikingRx-on-OAI/spx_records/raw/snr_top10_run_summary.txt", home);
  else
    snprintf(path, sizeof(path), "./snr_top10_run_summary.txt");

  FILE *fp = fopen(path, "a");
  if (!fp) {
    printf("[SPX_SNR_DUMP] fopen failed: %s\n", path);
    return;
  }

  fprintf(fp,
          "noise_power_db=%.3f snr_top10_db=%.6f used_windows=%d\n",
          noise_power_db,
          snr_top10_db,
          used_windows);
  fclose(fp);

  printf("[SPX_SNR_DUMP] path=%s noise_power_db=%.3f snr_top10_db=%.6f used_windows=%d\n",
         path,
         noise_power_db,
         snr_top10_db,
         used_windows);
}

/*
  Legacy study:
  The parameters are:
  gain&loss (decay, signal power, ...)
  either a fixed gain in dB, a target power in dBm or ACG (automatic control gain) to a target average
  => don't redo the AGC, as it was used in UE case, that must have a AGC inside the UE
  will be better to handle the "set_gain()" called by UE to apply it's gain (enable test of UE power loop)
  lin_amp = pow(10.0,.05*txpwr_dBm)/sqrt(nb_tx_antennas);
  a lot of operations in legacy, grouped in one simulation signal decay: txgain*decay*rxgain

  multi_path (auto convolution, ISI, ...)
  either we regenerate the channel (call again random_channel(desc,0)), or we keep it over subframes
  legacy: we regenerate each sub frame in UL, and each frame only in DL
*/
void rxAddInput(const c16_t *input_sig,
                cf_t *after_channel_sig,
                int rxAnt,
                channel_desc_t *channelDesc,
                int nbSamples,
                uint64_t TS,
                uint32_t CirSize,
                bool add_noise)
{
  static uint64_t last_TS = 0;

  /*
   * SPX SNR aggregation state
   * - reset automatically when noise_power_dB changes
   * - only rxAnt==0 participates
   * - collect a few valid windows, each window computes top10% SNR
   * - final dump = average of those window-level top10% SNRs
   */
  static int    spx_snr_dump_done = 0;
  static int    spx_snr_windows = 0;
  static double spx_snr_linear_acc = 0.0;
  static double spx_last_noise_power_db = 1e300;

  if (fabs(channelDesc->noise_power_dB - spx_last_noise_power_db) > 1e-12) {
    spx_snr_dump_done = 0;
    spx_snr_windows = 0;
    spx_snr_linear_acc = 0.0;
    spx_last_noise_power_db = channelDesc->noise_power_dB;
  }

  if ((channelDesc->sat_height > 0) && (channelDesc->enable_dynamic_delay || channelDesc->enable_dynamic_Doppler)) { // model for transparent satellite on circular orbit
    /* assumptions:
       - The Earth is spherical, the ground station is static, and that the Earth does not rotate.
       - An access or link is possible from the satellite to the ground station at all times.
       - The ground station is located at the North Pole (positive Zaxis), and the satellite starts from the initial elevation angle 0° in the second quadrant of the YZplane.
       - Satellite moves in the clockwise direction in its circular orbit.
    */
    const double radius_earth = 6377900; // m
    const double radius_sat = radius_earth + channelDesc->sat_height;
    const double GM_earth = 3.986e14; // m^3/s^2
    const double w_sat = sqrt(GM_earth / (radius_sat * radius_sat * radius_sat)); // rad/s

    // start_time and end_time are when the pos_sat_z == pos_gnb_z (elevation angle == 0 and 180 degree)
    const double start_phase = -acos(radius_earth / radius_sat); // SAT is just rising above the horizon
    const double end_phase = -start_phase; // SAT is just falling behind the horizon
    const double start_time = start_phase / w_sat; // in seconds
    const double end_time = end_phase / w_sat; // in seconds
    const uint64_t duration_samples = (end_time - start_time) * channelDesc->sampling_rate;
    const double t = start_time + ((TS - channelDesc->start_TS) % duration_samples) / channelDesc->sampling_rate;

    const double pos_sat_x = 0;
    const double pos_sat_y = radius_sat * sin(w_sat * t);
    const double pos_sat_z = radius_sat * cos(w_sat * t);

    const double vel_sat_x = 0;
    const double vel_sat_y =  w_sat * radius_sat * cos(w_sat * t);
    const double vel_sat_z = -w_sat * radius_sat * sin(w_sat * t);

    const double pos_ue_x = 0;
    const double pos_ue_y = 0;
    const double pos_ue_z = radius_earth;

    const double c = 299792458; // m/s

    if (channelDesc->is_uplink) {
      const double dir_ue_sat_x = pos_sat_x - pos_ue_x;
      const double dir_ue_sat_y = pos_sat_y - pos_ue_y;
      const double dir_ue_sat_z = pos_sat_z - pos_ue_z;

      const double dist_ue_sat = sqrt(dir_ue_sat_x * dir_ue_sat_x + dir_ue_sat_y * dir_ue_sat_y + dir_ue_sat_z * dir_ue_sat_z);
      const double vel_ue_sat = (vel_sat_x * dir_ue_sat_x + vel_sat_y * dir_ue_sat_y + vel_sat_z * dir_ue_sat_z) / dist_ue_sat;

      double dist_sat_gnb = 0;
      double vel_sat_gnb = 0;
      double acc_sat_gnb = 0;
      if (channelDesc->modelid == SAT_LEO_TRANS) {
        const double acc_sat_x = 0;
        const double acc_sat_y = -w_sat * w_sat * radius_sat * sin(w_sat * t);
        const double acc_sat_z = -w_sat * w_sat * radius_sat * cos(w_sat * t);

        const double pos_gnb_x = 0;
        const double pos_gnb_y = 0;
        const double pos_gnb_z = radius_earth;

        const double dir_sat_gnb_x = pos_gnb_x - pos_sat_x;
        const double dir_sat_gnb_y = pos_gnb_y - pos_sat_y;
        const double dir_sat_gnb_z = pos_gnb_z - pos_sat_z;

        dist_sat_gnb = sqrt(dir_sat_gnb_x * dir_sat_gnb_x + dir_sat_gnb_y * dir_sat_gnb_y + dir_sat_gnb_z * dir_sat_gnb_z);
        vel_sat_gnb = (vel_sat_x * dir_sat_gnb_x + vel_sat_y * dir_sat_gnb_y + vel_sat_z * dir_sat_gnb_z) / dist_sat_gnb;
        acc_sat_gnb = (acc_sat_x * dir_sat_gnb_x + acc_sat_y * dir_sat_gnb_y + acc_sat_z * dir_sat_gnb_z) / dist_sat_gnb;
      }

      const double prop_delay = (dist_ue_sat + dist_sat_gnb) / c;
      if (channelDesc->enable_dynamic_delay)
        channelDesc->channel_offset = prop_delay * channelDesc->sampling_rate;

      const double f_Doppler_shift_ue_sat = (-vel_ue_sat / c) * channelDesc->center_freq;
      if (channelDesc->enable_dynamic_Doppler)
        channelDesc->Doppler_phase_inc = 2 * M_PI * f_Doppler_shift_ue_sat / channelDesc->sampling_rate;

      if(TS - last_TS >= channelDesc->sampling_rate) {
        last_TS = TS;
        LOG_I(HW, "Satellite orbit: time %f s, Position = (%f, %f, %f), Velocity = (%f, %f, %f)\n", t, pos_sat_x, pos_sat_y, pos_sat_z, vel_sat_x, vel_sat_y, vel_sat_z);
        LOG_I(HW, "Uplink delay %f ms, Doppler shift UE->SAT %f kHz\n", prop_delay * 1000, f_Doppler_shift_ue_sat / 1000);
        LOG_I(HW, "Satellite velocity towards gNB: %f m/s, acceleration towards gNB: %f m/s²\n", vel_sat_gnb, acc_sat_gnb);
      }

      const int samples_per_subframe = channelDesc->sampling_rate / 1000;
      const int abs_subframe = TS / samples_per_subframe;
      if (abs_subframe % 10 == 0) { // update SIB19 information for the next frame
        gnb_sat_position_update_t sat_position = {
            .sfn = (abs_subframe / 10 + 1) % 1024,
            .subframe = 0,
            .delay = 2 * dist_sat_gnb / (c * 4.072e-9),
            .drift = 2 * -vel_sat_gnb / (c * 0.2e-9),
            .accel = 2 * acc_sat_gnb / (c * 0.2e-10),
            .position.X = pos_sat_x / 1.3,
            .position.Y = pos_sat_y / 1.3,
            .position.Z = pos_sat_z / 1.3,
            .velocity.X = vel_sat_x / 0.06,
            .velocity.Y = vel_sat_y / 0.06,
            .velocity.Z = vel_sat_z / 0.06,
        };
        nr_update_sib19(&sat_position);
      }
    } else {
      const double dir_sat_ue_x = pos_ue_x - pos_sat_x;
      const double dir_sat_ue_y = pos_ue_y - pos_sat_y;
      const double dir_sat_ue_z = pos_ue_z - pos_sat_z;

      const double dist_sat_ue = sqrt(dir_sat_ue_x * dir_sat_ue_x + dir_sat_ue_y * dir_sat_ue_y + dir_sat_ue_z * dir_sat_ue_z);
      const double vel_sat_ue = (vel_sat_x * dir_sat_ue_x + vel_sat_y * dir_sat_ue_y + vel_sat_z * dir_sat_ue_z) / dist_sat_ue;

      double dist_gnb_sat = 0;
      if (channelDesc->modelid == SAT_LEO_TRANS) {
        const double pos_gnb_x = 0;
        const double pos_gnb_y = 0;
        const double pos_gnb_z = radius_earth;

        const double dir_gnb_sat_x = pos_sat_x - pos_gnb_x;
        const double dir_gnb_sat_y = pos_sat_y - pos_gnb_y;
        const double dir_gnb_sat_z = pos_sat_z - pos_gnb_z;

        dist_gnb_sat = sqrt(dir_gnb_sat_x * dir_gnb_sat_x + dir_gnb_sat_y * dir_gnb_sat_y + dir_gnb_sat_z * dir_gnb_sat_z);
      }

      const double prop_delay = (dist_gnb_sat + dist_sat_ue) / c;
      if (channelDesc->enable_dynamic_delay)
        channelDesc->channel_offset = prop_delay * channelDesc->sampling_rate;

      const double f_Doppler_shift_sat_ue = (vel_sat_ue / (c - vel_sat_ue)) * channelDesc->center_freq;
      if (channelDesc->enable_dynamic_Doppler)
        channelDesc->Doppler_phase_inc = 2 * M_PI * f_Doppler_shift_sat_ue / channelDesc->sampling_rate;

      if(TS - last_TS >= channelDesc->sampling_rate) {
        last_TS = TS;
        LOG_I(HW, "Satellite orbit: time %f s, Position = (%f, %f, %f), Velocity = (%f, %f, %f)\n", t, pos_sat_x, pos_sat_y, pos_sat_z, vel_sat_x, vel_sat_y, vel_sat_z);
        LOG_I(HW, "Downlink delay %f ms, Doppler shift SAT->UE %f kHz\n", prop_delay * 1000, f_Doppler_shift_sat_ue / 1000);
      }
    }
  }

  // channelDesc->path_loss_dB should contain the total path gain
  const double pathLossLinear = pow(10, channelDesc->path_loss_dB / 20.0);
  // Energy in one sample to calibrate input noise
  const double noise_per_sample = add_noise ? pow(10, channelDesc->noise_power_dB / 10.0) * 256 : 0;
  const uint64_t dd = channelDesc->channel_offset;
  const int nbTx = channelDesc->nb_tx;
  double Doppler_phase_cur = channelDesc->Doppler_phase_cur[rxAnt];

  /*
   * We only collect SNR samples on rxAnt==0 and only until final dump is done.
   * That keeps the runtime much lower than "every slot always sort".
   */
  const int spx_collect_snr = (rxAnt == 0 && !spx_snr_dump_done);
  double snr_list[MAX_SNR_SAMPLES];
  int snr_count = 0;

  Doppler_phase_cur -= 2 * M_PI * round(Doppler_phase_cur / (2 * M_PI));

  for (int i = 0; i < nbSamples; i++) {
    cf_t *out_ptr = after_channel_sig + i;
    struct complexd rx_tmp = {0};

    for (int txAnt = 0; txAnt < nbTx; txAnt++) {
      const struct complexd *channelModel = channelDesc->ch[rxAnt + (txAnt * channelDesc->nb_rx)];

      for (int l = 0; l < (int)channelDesc->channel_length; l++) {
        const int idx = ((TS + i - l - dd) * nbTx + txAnt + CirSize) % CirSize;
        const struct complex16 tx16 = input_sig[idx];
        rx_tmp.r += tx16.r * channelModel[l].r - tx16.i * channelModel[l].i;
        rx_tmp.i += tx16.i * channelModel[l].r + tx16.r * channelModel[l].i;
      }
    }

    if (channelDesc->Doppler_phase_inc != 0.0) {
#ifdef CMPLX
      double complex in = CMPLX(rx_tmp.r, rx_tmp.i);
#else
      double complex in = rx_tmp.r + rx_tmp.i * I;
#endif
      double complex out = in * cexp(Doppler_phase_cur * I);
      rx_tmp.r = creal(out);
      rx_tmp.i = cimag(out);
      Doppler_phase_cur += channelDesc->Doppler_phase_inc;
    }

    const double sig_r = rx_tmp.r * pathLossLinear;
    const double sig_i = rx_tmp.i * pathLossLinear;
    const double noise_r = noise_per_sample * gaussZiggurat(0.0, 1.0);
    const double noise_i = noise_per_sample * gaussZiggurat(0.0, 1.0);

    out_ptr->r += sig_r + noise_r;
    out_ptr->i += sig_i + noise_i;

    if (spx_collect_snr && snr_count < MAX_SNR_SAMPLES) {
      const double sig_pow = sig_r * sig_r + sig_i * sig_i;
      const double noise_pow = noise_r * noise_r + noise_i * noise_i;

      if (sig_pow > 1e-6 && noise_pow > 0.0)
        snr_list[snr_count++] = sig_pow / noise_pow;
    }
  }

  channelDesc->Doppler_phase_cur[rxAnt] = Doppler_phase_cur;

  if (spx_collect_snr && snr_count >= MIN_VALID_SNR_SAMPLES) {
    qsort(snr_list, snr_count, sizeof(double), cmp_double_desc);

    int top_n = snr_count / 10;
    if (top_n < 1)
      top_n = 1;

    double sum = 0.0;
    for (int i = 0; i < top_n; i++)
      sum += snr_list[i];

    const double call_snr_linear = sum / top_n;
    spx_snr_linear_acc += call_snr_linear;
    spx_snr_windows++;

    if (spx_snr_windows >= TARGET_SNR_WINDOWS) {
      const double final_snr_linear = spx_snr_linear_acc / spx_snr_windows;
      const double final_snr_db = 10 * log10(final_snr_linear);

      printf("[SNR_TOP10_FINAL] noise_power_db=%.3f snr_db=%.6f used_windows=%d\n",
             channelDesc->noise_power_dB,
             final_snr_db,
             spx_snr_windows);

      spx_dump_snr_summary(channelDesc->noise_power_dB, final_snr_db, spx_snr_windows);
      spx_snr_dump_done = 1;
    }
  }

  if ((TS * nbTx) % CirSize + nbSamples <= CirSize)
    LOG_D(HW,"Input power %f, output power: %f, channel path loss %f, noise coeff: %f \n",
          10 * log10((double)signal_energy((int32_t *)&input_sig[(TS * nbTx) % CirSize], nbSamples)),
          10 * log10((double)signal_energy((int32_t *)after_channel_sig, nbSamples)),
          channelDesc->path_loss_dB,
          10 * log10(noise_per_sample));
}
