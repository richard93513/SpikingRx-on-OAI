# 2025/11/17 — Full Technical Report: OAI → SpikingRx Workflow (Final Integrated Version)

## 0. System Overview

This report documents the complete technical pipeline from building OAI gNB/UE → dumping PDSCH → Python 32×32 mapping → generating the input for SpikingRx.

Contents include:

- System environment  
- OAI compilation  
- rfsim connection  
- UE PDSCH dump (C code)  
- 600-RE → 32×32 mapping (Python)  
- Tensor verification (Python)  
- Technical analysis of all three code segments  
- (Images omitted per user request)


---

## 1. System Environment and OAI Initialization

### 1.1 Ubuntu Version  
Ubuntu 22.04.6 LTS (best compatibility with OAI)

### 1.2 Install OAI Dependencies

sudo apt update  
sudo apt install git build-essential cmake ninja-build \
    python3 python3-pip python3-venv \
    libboost-all-dev libsctp-dev libconfig++-dev

### 1.3 Download OAI

cd ~  
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git  
cd openairinterface5g

### 1.4 Compile OAI (gNB + nrUE)

Clear cache:

cd ~/openairinterface5g/cmake_targets/ran_build/build  
rm -rf CMakeCache.txt CMakeFiles

Compile:

cd ~/openairinterface5g/cmake_targets  
sudo ./build_oai --gNB --nrUE -w SIMU


---

## 2. rfsim Startup and Connection

### 2.1 Start gNB

cd ~/openairinterface5g/cmake_targets/ran_build/build  
sudo ./nr-softmodem --rfsim --sa \
  -O ci-scripts/conf_files/gnb.sa.band78.106prb.rfsim.conf

### 2.2 Start UE (new terminal)

cd ~/openairinterface5g/cmake_targets/ran_build/build  
sudo ./nr-uesoftmodem --rfsim --sa

### 2.3 Expected Successful Connection Output

You should see:

- RSRP / SNR  
- HARQ  
- RNTI established  
- Slot incrementing  
- UE decoding PDSCH successfully


---

## 3. Full Dump Workflow: rxdataF_ext (Slot-level)

# A. PDSCH Dump (UE-side C Program)

## A.1 Source Code

(if code block appears in the next section, it is intentional and will be delivered in PART 2)
