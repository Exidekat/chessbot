#!/bin/bash
# Monitor USB connection stability for WBC-0E01 camera

echo "Monitoring USB device activity (Bus 003 Device 017)..."
echo "Press Ctrl+C to stop"
echo ""

# Watch dmesg for USB events
sudo dmesg -w | grep -i --line-buffered -E "(usb|uvc|video)"
