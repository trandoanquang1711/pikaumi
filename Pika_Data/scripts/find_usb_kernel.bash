#!/bin/bash
# Original code from https://github.com/JHyoonirl/pika_ros2/blob/main/scripts/find_addr.bash

# --- Terminal colors ---
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${GREEN}=== Pika Final Physical Address Scanner (No-Freeze) ===${NC}"
echo "------------------------------------------------"

# 1. Port Scan
echo -e "${CYAN}[1] Serial Port (ttyUSB)${NC}"
for dev in /dev/ttyUSB*; do
    [ -e "$dev" ] || continue # Skip if no ttyUSB device is present
    
    # Step 1: Resolve full sysfs physical device path
    # Example: /sys/devices/.../usb3/3-2/3-2.2/3-2.2.1/3-2.2.1.4/3-2.2.1.4:1.0/ttyUSB0
    real_path=$(readlink -f /sys/class/tty/$(basename "$dev")/device)
    
    # Step 2: The parent of the interface (:1.0) directory gives the USB kernel port address.
    addr=$(basename $(dirname "$real_path"))
    
    echo -e "  Device : ${YELLOW}$dev${NC}"
    echo -e "  Address: ${GREEN}$addr${NC}"
    echo "------------------------------------------------"
done

# 2. Fisheye Camera Scan
echo -e "${CYAN}[2] Fisheye Camera${NC}"
for dev in /sys/class/video4linux/video*; do
    [ -e "$dev" ] || continue
    vid_name=$(basename "$dev")
    
    # Match USB vendor ID 1bcf (Pika fisheye camera vendor)
    if [ -f "$dev/device/../idVendor" ] && [ "$(cat "$dev/device/../idVendor")" == "1bcf" ]; then
        real_path=$(readlink -f "$dev/device")
        
        # Video sysfs path layout may or may not end with an interface suffix (:1.0)
        if [[ "$real_path" == *":"* ]]; then
            addr=$(basename $(dirname "$real_path"))
        else
            addr=$(basename "$real_path")
        fi
        
        echo -e "  Device : ${YELLOW}/dev/$vid_name${NC}"
        echo -e "  Address: ${GREEN}$addr${NC}"
        echo "------------------------------------------------"
    fi
done

echo -e "${GREEN}Scanning complete.${NC}"