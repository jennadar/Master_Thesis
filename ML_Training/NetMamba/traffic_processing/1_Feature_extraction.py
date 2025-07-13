#1. Preprocessing step
from scapy.all import rdpcap, IP, TCP, UDP, Raw
import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone

def extract_packet_flow_features(pcap_file, output_csv):
    packets = rdpcap(pcap_file)
    data = []
    flows = defaultdict(list)

    for pkt in packets:
        if IP in pkt:
            ts = float(pkt.time)
            formatted_time = datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = pkt[IP].proto
            pkt_len = len(pkt)
            raw_len = len(pkt[Raw].load) if Raw in pkt else 0

            src_port = pkt.sport if (TCP in pkt or UDP in pkt) else 0
            dst_port = pkt.dport if (TCP in pkt or UDP in pkt) else 0
            tcp_flags = pkt[TCP].flags if TCP in pkt else 'N/A'

            flow_key = (src_ip, dst_ip, proto, src_port, dst_port)

            if flows[flow_key]:
                last_ts = flows[flow_key][-1]["ts"]
                iat = ts - last_ts
            else:
                iat = 0  

            flows[flow_key].append({"ts": ts, "pkt_len": pkt_len})

            flow_duration = ts - flows[flow_key][0]["ts"] if len(flows[flow_key]) > 1 else 0
            total_packets = len(flows[flow_key])
            total_bytes = sum(p["pkt_len"] for p in flows[flow_key])
            packets_per_second = total_packets / flow_duration if flow_duration > 0 else 0
            bytes_per_second = total_bytes / flow_duration if flow_duration > 0 else 0
            mean_pkt_size = total_bytes / total_packets if total_packets > 0 else 0
            mean_iat = sum(flows[flow_key][j]["ts"] - flows[flow_key][j-1]["ts"] 
                           for j in range(1, total_packets)) / (total_packets - 1) if total_packets > 1 else 0

            data.append({
                "ts": ts,
                "formatted_time": formatted_time,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "proto": proto,
                "src_port": src_port,
                "dst_port": dst_port,
                "packet_length": pkt_len,
                "raw_len": raw_len,
                "inter_packet_time": iat,
                "flow_duration": flow_duration,
                "total_packets_in_flow": total_packets,
                "total_bytes_in_flow": total_bytes,
                "mean_packet_size": mean_pkt_size,
                "mean_inter_arrival_time": mean_iat,
                "packets_per_second": packets_per_second,
                "bytes_per_second": bytes_per_second,
                "tcp_flags": tcp_flags
            })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

# Example Usage
pcap_file = "D:/jenny/Documents/FAUS_Study/Thesis/iot_23_datasets_full/IoTScenarios/CTU-IoT-Malware-Capture-36-1/2018-12-21-13-36-41-192.168.1.198.pcap"
output_csv = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-36-1/2018-12-21-13-36-41-192.168.1.198_packet_flow_features.csv"
extract_packet_flow_features(pcap_file, output_csv)
