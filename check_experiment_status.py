#!/usr/bin/env python3
"""
Real-time experiment status checker
Monitors running experiments and provides status updates
"""

import os
import time
import psutil
import json
from pathlib import Path
from datetime import datetime, timedelta


class ExperimentMonitor:
    def __init__(self):
        self.results_dir = Path("quick_tuning_results")
        
    def find_running_experiments(self):
        """Find all running experiment processes"""
        
        running_experiments = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'run_enhanced.py' in ' '.join(cmdline):
                    running_experiments.append({
                        'pid': proc.info['pid'],
                        'cmdline': ' '.join(cmdline),
                        'start_time': datetime.fromtimestamp(proc.info['create_time']),
                        'cpu_percent': proc.cpu_percent(),
                        'memory_percent': proc.memory_percent(),
                        'memory_mb': proc.memory_info().rss / 1024 / 1024
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return running_experiments
    
    def check_latest_results(self):
        """Check the latest results from JSON files"""
        
        results_files = [
            "quick_tuning_results/quick_all_results.json",
            "quick_tuning_results/quick_3dshapes_results.json",
            "quick_tuning_results/quick_dsprites_results.json"
        ]
        
        latest_results = {}
        
        for file_path in results_files:
            path = Path(file_path)
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    
                    # Get file modification time
                    mtime = datetime.fromtimestamp(path.stat().st_mtime)
                    
                    # Count successful experiments
                    successful = [r for r in data if r.get('success', False)]
                    failed = [r for r in data if not r.get('success', False)]
                    
                    latest_results[file_path] = {
                        'last_modified': mtime,
                        'total_experiments': len(data),
                        'successful': len(successful),
                        'failed': len(failed),
                        'latest_dci_d': None
                    }
                    
                    # Get latest DCI-D score
                    if successful:
                        latest_dci_d = max([r.get('dci_disentanglement', 0) for r in successful if r.get('dci_disentanglement')])
                        latest_results[file_path]['latest_dci_d'] = latest_dci_d
                        
                except Exception as e:
                    latest_results[file_path] = {'error': str(e)}
        
        return latest_results
    
    def get_system_status(self):
        """Get system resource status"""
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / 1024 / 1024 / 1024,
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / 1024 / 1024 / 1024
        }
    
    def print_status_report(self):
        """Print comprehensive status report"""
        
        print("\n" + "="*80)
        print("EXPERIMENT STATUS REPORT")
        print("="*80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check running experiments
        running_experiments = self.find_running_experiments()
        
        if running_experiments:
            print(f"\n🔄 RUNNING EXPERIMENTS ({len(running_experiments)})")
            print("-" * 50)
            
            for exp in running_experiments:
                runtime = datetime.now() - exp['start_time']
                print(f"PID: {exp['pid']}")
                print(f"Command: {exp['cmdline'][:80]}...")
                print(f"Runtime: {runtime}")
                print(f"CPU: {exp['cpu_percent']:.1f}% | Memory: {exp['memory_mb']:.1f} MB")
                print()
        else:
            print("\n⏹️  NO RUNNING EXPERIMENTS")
        
        # Check latest results
        latest_results = self.check_latest_results()
        
        if latest_results:
            print("📊 LATEST RESULTS")
            print("-" * 30)
            
            for file_path, info in latest_results.items():
                if 'error' in info:
                    print(f"❌ {file_path}: {info['error']}")
                    continue
                
                print(f"📁 {file_path}")
                print(f"   Last modified: {info['last_modified'].strftime('%H:%M:%S')}")
                print(f"   Total experiments: {info['total_experiments']}")
                print(f"   Successful: {info['successful']} | Failed: {info['failed']}")
                
                if info['latest_dci_d'] is not None:
                    print(f"   Best DCI-D: {info['latest_dci_d']:.4f}")
                print()
        
        # System status
        system_status = self.get_system_status()
        
        print("💻 SYSTEM STATUS")
        print("-" * 20)
        print(f"CPU Usage: {system_status['cpu_percent']:.1f}%")
        print(f"Memory Usage: {system_status['memory_percent']:.1f}% ({system_status['memory_available_gb']:.1f} GB available)")
        print(f"Disk Usage: {system_status['disk_percent']:.1f}% ({system_status['disk_free_gb']:.1f} GB free)")
        
        print("\n" + "="*80)
    
    def monitor_continuously(self, interval: int = 30):
        """Monitor continuously with specified interval"""
        
        print(f"🔍 Starting continuous monitoring (checking every {interval} seconds)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                self.print_status_report()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor experiment status")
    parser.add_argument("--continuous", "-c", action="store_true", 
                       help="Monitor continuously")
    parser.add_argument("--interval", "-i", type=int, default=30,
                       help="Check interval in seconds (default: 30)")
    
    args = parser.parse_args()
    
    monitor = ExperimentMonitor()
    
    if args.continuous:
        monitor.monitor_continuously(args.interval)
    else:
        monitor.print_status_report()


if __name__ == "__main__":
    main() 