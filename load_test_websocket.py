#!/usr/bin/env python3
"""
WebSocket Load Testing Script
Tests real-time EKYC API performance under load
"""

import asyncio
import websockets
import json
import time
import base64
import argparse
from datetime import datetime
import statistics
from pathlib import Path
import numpy as np
import cv2
from typing import List, Dict, Any
import sys

class WebSocketLoadTester:
    def __init__(self, server_url: str, num_clients: int = 10):
        self.server_url = server_url
        self.num_clients = num_clients
        self.results = []
        self.errors = []
        self.latencies = []
        
    async def create_dummy_frame(self) -> str:
        """Create a dummy JPEG frame for testing"""
        # Create a 224x224 random image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Encode as JPEG
        success, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if success:
            # Encode as base64
            b64_frame = base64.b64encode(encoded.tobytes()).decode('utf-8')
            return b64_frame
        else:
            raise Exception("Failed to encode dummy frame")
    
    async def client_task(self, client_id: int, frames_per_client: int = 50):
        """Run a single client connection"""
        try:
            uri = f"{self.server_url}/ws/ekyc-stream/load_test_client_{client_id}"
            
            async with websockets.connect(uri, ping_interval=10, ping_timeout=5) as websocket:
                print(f"[Client {client_id}] Connected")
                
                for frame_idx in range(frames_per_client):
                    try:
                        # Create dummy frame
                        frame_data = await self.create_dummy_frame()
                        
                        # Send frame
                        send_time = time.time()
                        message = {
                            "type": "frame",
                            "data": frame_data,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        await websocket.send(json.dumps(message))
                        
                        # Wait for inference result
                        response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        recv_time = time.time()
                        
                        # Calculate latency
                        latency = (recv_time - send_time) * 1000  # Convert to ms
                        self.latencies.append(latency)
                        
                        # Parse response
                        result = json.loads(response)
                        
                        self.results.append({
                            'client_id': client_id,
                            'frame_idx': frame_idx,
                            'latency_ms': latency,
                            'result': result
                        })
                        
                        if frame_idx % 10 == 0:
                            print(f"[Client {client_id}] Sent {frame_idx} frames, latency: {latency:.1f}ms")
                        
                    except asyncio.TimeoutError:
                        error_msg = f"Client {client_id} timeout at frame {frame_idx}"
                        print(f"ERROR: {error_msg}")
                        self.errors.append(error_msg)
                        break
                    except Exception as e:
                        error_msg = f"Client {client_id} error at frame {frame_idx}: {str(e)}"
                        print(f"ERROR: {error_msg}")
                        self.errors.append(error_msg)
                        break
                
                print(f"[Client {client_id}] Completed")
                
        except Exception as e:
            error_msg = f"Client {client_id} connection failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            self.errors.append(error_msg)
    
    async def run_load_test(self, frames_per_client: int = 50, duration_seconds: int = 60):
        """Run load test with multiple concurrent clients"""
        print(f"\n{'='*60}")
        print(f"Starting WebSocket Load Test")
        print(f"{'='*60}")
        print(f"Server URL: {self.server_url}")
        print(f"Number of clients: {self.num_clients}")
        print(f"Frames per client: {frames_per_client}")
        print(f"Max duration: {duration_seconds}s")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Create tasks for all clients
        tasks = [
            self.client_task(i, frames_per_client) 
            for i in range(self.num_clients)
        ]
        
        # Run tasks with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=duration_seconds
            )
        except asyncio.TimeoutError:
            print(f"\nTest reached maximum duration of {duration_seconds}s")
        
        elapsed_time = time.time() - start_time
        
        # Generate report
        self.generate_report(elapsed_time)
    
    def generate_report(self, elapsed_time: float):
        """Generate performance report"""
        print(f"\n{'='*60}")
        print(f"Load Test Results")
        print(f"{'='*60}\n")
        
        print(f"Duration: {elapsed_time:.2f}s")
        print(f"Total frames sent: {len(self.results)}")
        print(f"Total errors: {len(self.errors)}")
        print(f"Success rate: {(len(self.results) / (len(self.results) + len(self.errors)) * 100):.1f}%")
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")
        
        if self.latencies:
            print(f"\nLatency Statistics (ms):")
            print(f"  Min:        {min(self.latencies):.2f}")
            print(f"  Max:        {max(self.latencies):.2f}")
            print(f"  Mean:       {statistics.mean(self.latencies):.2f}")
            print(f"  Median:     {statistics.median(self.latencies):.2f}")
            print(f"  StdDev:     {statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0:.2f}")
            print(f"  P95:        {np.percentile(self.latencies, 95):.2f}")
            print(f"  P99:        {np.percentile(self.latencies, 99):.2f}")
            
            # Throughput
            frames_per_second = len(self.results) / elapsed_time
            print(f"\nThroughput:")
            print(f"  Frames/sec: {frames_per_second:.2f}")
            print(f"  Inferences/sec: {frames_per_second:.2f}")  # One inference per frame
        
        # Connection statistics
        if self.results:
            unique_clients = len(set(r['client_id'] for r in self.results))
            frames_per_client = len(self.results) / unique_clients if unique_clients > 0 else 0
            print(f"\nConnection Statistics:")
            print(f"  Active clients: {unique_clients}")
            print(f"  Frames per client: {frames_per_client:.1f}")
        
        # Model performance
        labels = [r['result'].get('label', 'UNKNOWN') for r in self.results if 'result' in r]
        if labels:
            print(f"\nModel Predictions:")
            print(f"  REAL: {labels.count('REAL')}")
            print(f"  SPOOF: {labels.count('SPOOF')}")
            print(f"  UNKNOWN: {labels.count('UNKNOWN')}")
        
        # Recommendations
        print(f"\n{'='*60}")
        print(f"Recommendations")
        print(f"{'='*60}")
        
        avg_latency = statistics.mean(self.latencies) if self.latencies else 0
        
        if avg_latency < 300:
            print("✓ Latency is excellent (<300ms)")
        elif avg_latency < 500:
            print("✓ Latency is good (300-500ms)")
        elif avg_latency < 1000:
            print("⚠ Latency is acceptable (500-1000ms)")
        else:
            print("✗ Latency needs improvement (>1000ms)")
            print("  - Consider GPU acceleration")
            print("  - Reduce model size or quantize")
            print("  - Scale horizontally (multiple servers)")
        
        error_rate = len(self.errors) / (len(self.results) + len(self.errors))
        if error_rate > 0.01:
            print(f"\n✗ High error rate ({error_rate*100:.1f}%)")
            print("  - Check server stability")
            print("  - Increase server resources")
            print("  - Review error logs")
        elif error_rate == 0:
            print("\n✓ Zero errors - System is stable")
        
        print(f"\n{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(description='WebSocket Load Test for EKYC API')
    parser.add_argument('--server', default='ws://localhost:8000', 
                        help='Server URL (default: ws://localhost:8000)')
    parser.add_argument('--clients', type=int, default=5,
                        help='Number of concurrent clients (default: 5)')
    parser.add_argument('--frames', type=int, default=50,
                        help='Frames per client (default: 50)')
    parser.add_argument('--duration', type=int, default=300,
                        help='Max test duration in seconds (default: 300)')
    
    args = parser.parse_args()
    
    tester = WebSocketLoadTester(args.server, args.clients)
    await tester.run_load_test(args.frames, args.duration)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
