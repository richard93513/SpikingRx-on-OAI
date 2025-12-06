#ifndef COMMON_LIB_H
#define COMMON_LIB_H

// Minimal stub for LDPC standalone decoder
// Only defines types required by LDPC includes

typedef struct {
    int dummy;
} openair0_rf_config_t;

typedef struct {
    void *rxp[8];
    void *txp[8];
} openair0_rx_tx_t;

#endif

