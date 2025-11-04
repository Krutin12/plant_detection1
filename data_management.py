"""
Data Management Module
=====================
Handles system data backup, cleanup, and maintenance operations.
"""

import json
import shutil
import os
from datetime import datetime, timedelta
from pathlib import Path
import zipfile

class DataManagementModule:
    """System data management and maintenance"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.backup_path = self.base_path / 'backups'
        self.backup_path.mkdir(exist_ok=True)
        
        self.data_folders = [
            'detection_history',
            'treatment_history', 
            'fertilizer_data',
            'user_profiles',
            'analytics',
            'models'
        ]
    
    def create_system_backup(self):
        """Create complete system backup"""
        print("\n--- CREATING SYSTEM BACKUP ---")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"system_backup_{timestamp}"
        backup_folder = self.backup_path / backup_name
        backup_folder.mkdir(exist_ok=True)
        
        backed_up_items = 0
        
        # Backup each data folder
        for folder_name in self.data_folders:
            source_folder = self.base_path / folder_name
            
            if source_folder.exists():
                dest_folder = backup_folder / folder_name
                try:
                    shutil.copytree(source_folder, dest_folder)
                    backed_up_items += 1
                    print(f"Backed up: {folder_name}")
                except Exception as e:
                    print(f"Error backing up {folder_name}: {e}")
        
        # Create backup manifest
        manifest = {
            'backup_date': datetime.now().isoformat(),
            'items_backed_up': backed_up_items,
            'folders': self.data_folders,
            'backup_size_mb': self.get_folder_size(backup_folder)
        }
        
        manifest_file = backup_folder / 'backup_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create ZIP archive
        zip_file = self.backup_path / f"{backup_name}.zip"
        self.create_zip_archive(backup_folder, zip_file)
        
        # Remove uncompressed backup folder
        shutil.rmtree(backup_folder)
        
        print(f"\nBackup completed: {zip_file}")
        print(f"Items backed up: {backed_up_items}")
        print(f"Backup size: {manifest['backup_size_mb']:.1f} MB")
        
        return zip_file
    
    def create_zip_archive(self, source_folder, zip_file):
        """Create ZIP archive from folder"""
        try:
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in source_folder.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(source_folder)
                        zipf.write(file_path, arcname)
        except Exception as e:
            print(f"Error creating ZIP archive: {e}")
    
    def restore_from_backup(self):
        """Restore system from backup"""
        print("\n--- RESTORE FROM BACKUP ---")
        
        # List available backups
        backup_files = list(self.backup_path.glob('*.zip'))
        if not backup_files:
            print("No backup files found.")
            return
        
        print("Available backups:")
        for i, backup_file in enumerate(sorted(backup_files, reverse=True), 1):
            # Extract date from filename
            filename = backup_file.stem
            if 'backup_' in filename:
                date_part = filename.split('_')[-2:]
                if len(date_part) == 2:
                    date_str = f"{date_part[0]}_{date_part[1]}"
                    try:
                        backup_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                        print(f"{i}. {filename} ({backup_date.strftime('%Y-%m-%d %H:%M')})")
                    except:
                        print(f"{i}. {filename}")
                else:
                    print(f"{i}. {filename}")
        
        try:
            choice = int(input("Select backup to restore (number): ")) - 1
            if 0 <= choice < len(backup_files):
                selected_backup = sorted(backup_files, reverse=True)[choice]
                
                # Confirm restoration
                print(f"\nSelected backup: {selected_backup.name}")
                confirm = input("This will overwrite current data. Continue? (type 'RESTORE' to confirm): ")
                
                if confirm == 'RESTORE':
                    self.perform_restoration(selected_backup)
                else:
                    print("Restoration cancelled.")
            else:
                print("Invalid selection.")
                
        except ValueError:
            print("Invalid input.")
    
    def perform_restoration(self, backup_file):
        """Perform the actual restoration process"""
        try:
            # Create temporary extraction folder
            temp_folder = self.backup_path / 'temp_restore'
            temp_folder.mkdir(exist_ok=True)
            
            # Extract backup
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                zipf.extractall(temp_folder)
            
            restored_items = 0
            
            # Restore each folder
            for folder_name in self.data_folders:
                source_folder = temp_folder / folder_name
                dest_folder = self.base_path / folder_name
                
                if source_folder.exists():
                    # Remove existing folder if it exists
                    if dest_folder.exists():
                        shutil.rmtree(dest_folder)
                    
                    # Copy from backup
                    shutil.copytree(source_folder, dest_folder)
                    restored_items += 1
                    print(f"Restored: {folder_name}")
            
            # Clean up temp folder
            shutil.rmtree(temp_folder)
            
            print(f"\nRestoration completed!")
            print(f"Items restored: {restored_items}")
            
        except Exception as e:
            print(f"Error during restoration: {e}")
    
    def clean_system_data(self):
        """Clean up system data"""
        print("\n--- SYSTEM DATA CLEANUP ---")
        print("1. Clean old backups (>30 days)")
        print("2. Clean temporary files")
        print("3. Clean empty folders")
        print("4. Full cleanup (all above)")
        
        choice = input("Select cleanup option (1-4): ").strip()
        
        cleaned_items = 0
        
        if choice in ['1', '4']:
            cleaned_items += self.clean_old_backups()
        
        if choice in ['2', '4']:
            cleaned_items += self.clean_temp_files()
        
        if choice in ['3', '4']:
            cleaned_items += self.clean_empty_folders()
        
        print(f"\nCleanup completed. {cleaned_items} items cleaned.")
    
    def clean_old_backups(self):
        """Clean backups older than 30 days"""
        cutoff_date = datetime.now().timestamp() - (30 * 24 * 60 * 60)
        cleaned_count = 0
        
        for backup_file in self.backup_path.glob('*.zip'):
            if backup_file.stat().st_mtime < cutoff_date:
                try:
                    backup_file.unlink()
                    cleaned_count += 1
                    print(f"Deleted old backup: {backup_file.name}")
                except Exception as e:
                    print(f"Error deleting {backup_file.name}: {e}")
        
        return cleaned_count
    
    def clean_temp_files(self):
        """Clean temporary files"""
        temp_patterns = ['*.tmp', '*.temp', '*~', '.DS_Store']
        cleaned_count = 0
        
        for pattern in temp_patterns:
            for temp_file in self.base_path.rglob(pattern):
                try:
                    temp_file.unlink()
                    cleaned_count += 1
                    print(f"Deleted temp file: {temp_file.name}")
                except Exception as e:
                    print(f"Error deleting {temp_file.name}: {e}")
        
        return cleaned_count
    
    def clean_empty_folders(self):
        """Remove empty folders"""
        cleaned_count = 0
        
        for folder in self.base_path.rglob('*'):
            if folder.is_dir() and not any(folder.iterdir()):
                try:
                    folder.rmdir()
                    cleaned_count += 1
                    print(f"Removed empty folder: {folder.name}")
                except Exception as e:
                    print(f"Error removing {folder.name}: {e}")
        
        return cleaned_count
    
    def get_system_statistics(self):
        """Get comprehensive system statistics"""
        stats = {
            'total_size_mb': 0,
            'folder_stats': {},
            'file_counts': {},
            'last_modified': {}
        }
        
        for folder_name in self.data_folders:
            folder_path = self.base_path / folder_name
            
            if folder_path.exists():
                folder_size = self.get_folder_size(folder_path)
                file_count = len(list(folder_path.rglob('*'))) 
                
                # Get last modified time
                latest_time = 0
                for file_path in folder_path.rglob('*'):
                    if file_path.is_file():
                        latest_time = max(latest_time, file_path.stat().st_mtime)
                
                stats['folder_stats'][folder_name] = folder_size
                stats['file_counts'][folder_name] = file_count
                stats['last_modified'][folder_name] = datetime.fromtimestamp(latest_time) if latest_time > 0 else None
                stats['total_size_mb'] += folder_size
        
        return stats
    
    def get_folder_size(self, folder_path):
        """Calculate folder size in MB"""
        total_size = 0
        try:
            for file_path in folder_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            print(f"Error calculating size for {folder_path}: {e}")
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def export_system_info(self):
        """Export system information"""
        print("\n--- EXPORT SYSTEM INFORMATION ---")
        
        stats = self.get_system_statistics()
        
        # Create system info report
        report = {
            'export_date': datetime.now().isoformat(),
            'system_statistics': stats,
            'data_folders': self.data_folders,
            'backup_location': str(self.backup_path),
            'available_backups': [f.name for f in self.backup_path.glob('*.zip')]
        }
        
        # Save report
        report_file = self.base_path / f'system_info_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"System information exported to: {report_file}")
            
        except Exception as e:
            print(f"Error exporting system info: {e}")
    
    def view_system_status(self):
        """Display current system status"""
        print("\n--- SYSTEM STATUS ---")
        
        stats = self.get_system_statistics()
        
        print(f"Total System Size: {stats['total_size_mb']:.1f} MB")
        
        print(f"\nFolder Statistics:")
        for folder_name, size in stats['folder_stats'].items():
            file_count = stats['file_counts'].get(folder_name, 0)
            last_modified = stats['last_modified'].get(folder_name)
            
            print(f"  {folder_name}:")
            print(f"    Size: {size:.1f} MB")
            print(f"    Files: {file_count}")
            print(f"    Last Modified: {last_modified.strftime('%Y-%m-%d %H:%M') if last_modified else 'Never'}")
        
        # Backup information
        backup_files = list(self.backup_path.glob('*.zip'))
        print(f"\nBackup Information:")
        print(f"  Available Backups: {len(backup_files)}")
        
        if backup_files:
            latest_backup = max(backup_files, key=os.path.getctime)
            backup_date = datetime.fromtimestamp(latest_backup.stat().st_mtime)
            print(f"  Latest Backup: {backup_date.strftime('%Y-%m-%d %H:%M')}")
        
        # System health
        print(f"\nSystem Health:")
        health_score = self.calculate_health_score(stats)
        print(f"  Health Score: {health_score}/100")
        
        if health_score < 80:
            print("  Recommendations:")
            if stats['total_size_mb'] > 1000:
                print("    - Consider cleaning old data")
            if len(backup_files) == 0:
                print("    - Create a system backup")
            if len(backup_files) > 10:
                print("    - Clean old backups")
    
    def calculate_health_score(self, stats):
        """Calculate system health score (0-100)"""
        score = 100
        
        # Deduct for large system size
        if stats['total_size_mb'] > 1000:
            score -= 10
        
        # Deduct if no recent activity
        recent_activity = False
        cutoff_time = datetime.now() - timedelta(days=7)
        
        for last_modified in stats['last_modified'].values():
            if last_modified and last_modified > cutoff_time:
                recent_activity = True
                break
        
        if not recent_activity:
            score -= 20
        
        # Check backup status
        backup_files = list(self.backup_path.glob('*.zip'))
        if len(backup_files) == 0:
            score -= 15
        elif len(backup_files) > 10:
            score -= 5
        
        return max(0, score)
    
    def run(self):
        """Main data management interface"""
        while True:
            print(f"\n--- DATA MANAGEMENT MODULE ---")
            print("1. View System Status")
            print("2. Create System Backup")
            print("3. Restore from Backup")
            print("4. Clean System Data")
            print("5. Export System Information")
            print("6. Database Maintenance")
            print("7. Return to Main Menu")
            
            choice = input("Select option (1-7): ").strip()
            
            if choice == '1':
                self.view_system_status()
                
            elif choice == '2':
                self.create_system_backup()
                
            elif choice == '3':
                self.restore_from_backup()
                
            elif choice == '4':
                self.clean_system_data()
                
            elif choice == '5':
                self.export_system_info()
                
            elif choice == '6':
                self.database_maintenance()
                
            elif choice == '7':
                break
                
            else:
                print("Invalid choice. Please select 1-7.")
    
    def database_maintenance(self):
        """Perform database maintenance tasks"""
        print("\n--- DATABASE MAINTENANCE ---")
        print("1. Verify Data Integrity")
        print("2. Optimize Data Storage")
        print("3. Rebuild Indexes")
        print("4. Check for Duplicates")
        
        choice = input("Select maintenance task (1-4): ").strip()
        
        if choice == '1':
            self.verify_data_integrity()
        elif choice == '2':
            self.optimize_data_storage()
        elif choice == '3':
            self.rebuild_indexes()
        elif choice == '4':
            self.check_for_duplicates()
        else:
            print("Invalid choice.")
    
    def verify_data_integrity(self):
        """Verify integrity of data files"""
        print("Verifying data integrity...")
        
        issues_found = 0
        
        for folder_name in self.data_folders:
            folder_path = self.base_path / folder_name
            
            if folder_path.exists():
                for json_file in folder_path.glob('*.json'):
                    try:
                        with open(json_file, 'r') as f:
                            json.load(f)
                        print(f"✓ {json_file.name}")
                    except json.JSONDecodeError:
                        print(f"✗ {json_file.name} - Invalid JSON")
                        issues_found += 1
                    except Exception as e:
                        print(f"✗ {json_file.name} - {str(e)}")
                        issues_found += 1
        
        print(f"\nIntegrity check completed. Issues found: {issues_found}")
    
    def optimize_data_storage(self):
        """Optimize data storage by compacting files"""
        print("Optimizing data storage...")
        
        optimized_files = 0
        
        for folder_name in self.data_folders:
            folder_path = self.base_path / folder_name
            
            if folder_path.exists():
                for json_file in folder_path.glob('*.json'):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        # Rewrite with compact format
                        with open(json_file, 'w') as f:
                            json.dump(data, f, separators=(',', ':'))
                        
                        optimized_files += 1
                        
                    except Exception as e:
                        print(f"Error optimizing {json_file.name}: {e}")
        
        print(f"Optimization completed. {optimized_files} files optimized.")
    
    def rebuild_indexes(self):
        """Rebuild data indexes for faster access"""
        print("Rebuilding data indexes...")
        
        # This would typically rebuild database indexes
        # For JSON files, we can create summary indexes
        
        index_data = {
            'detection_summary': self.create_detection_index(),
            'treatment_summary': self.create_treatment_index(),
            'last_rebuilt': datetime.now().isoformat()
        }
        
        index_file = self.base_path / 'system_index.json'
        
        try:
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            print("Indexes rebuilt successfully.")
            
        except Exception as e:
            print(f"Error rebuilding indexes: {e}")
    
    def create_detection_index(self):
        """Create detection data index"""
        detection_file = self.base_path / 'detection_history' / 'detection_history.json'
        
        if detection_file.exists():
            try:
                with open(detection_file, 'r') as f:
                    data = json.load(f)
                
                return {
                    'total_records': len(data),
                    'date_range': {
                        'start': min(record['timestamp'] for record in data) if data else None,
                        'end': max(record['timestamp'] for record in data) if data else None
                    },
                    'diseases': list(set(record.get('predicted_disease', 'Unknown') for record in data))
                }
            except:
                pass
        
        return {'total_records': 0}
    
    def create_treatment_index(self):
        """Create treatment data index"""
        treatment_file = self.base_path / 'treatment_history' / 'treatment_history.json'
        
        if treatment_file.exists():
            try:
                with open(treatment_file, 'r') as f:
                    data = json.load(f)
                
                return {
                    'total_records': len(data),
                    'status_distribution': self.get_status_counts(data),
                    'treatment_types': list(set(record.get('treatment_type', 'Unknown') for record in data))
                }
            except:
                pass
        
        return {'total_records': 0}
    
    def get_status_counts(self, data):
        """Get status counts from treatment data"""
        status_counts = {}
        for record in data:
            status = record.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts
    
    def check_for_duplicates(self):
        """Check for duplicate records across data files"""
        print("Checking for duplicate records...")
        
        duplicates_found = 0
        
        # Check detection history
        detection_file = self.base_path / 'detection_history' / 'detection_history.json'
        if detection_file.exists():
            duplicates_found += self.find_duplicates_in_file(detection_file, 'Detection')
        
        # Check treatment history
        treatment_file = self.base_path / 'treatment_history' / 'treatment_history.json'
        if treatment_file.exists():
            duplicates_found += self.find_duplicates_in_file(treatment_file, 'Treatment')
        
        print(f"Duplicate check completed. {duplicates_found} duplicates found.")
    
    def find_duplicates_in_file(self, file_path, data_type):
        """Find duplicates in a specific file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            seen_records = set()
            duplicates = 0
            
            for record in data:
                # Create a hash of key fields
                key_fields = str(sorted(record.items()))
                
                if key_fields in seen_records:
                    print(f"Duplicate {data_type} record found: ID {record.get('id', 'Unknown')}")
                    duplicates += 1
                else:
                    seen_records.add(key_fields)
            
            return duplicates
            
        except Exception as e:
            print(f"Error checking duplicates in {file_path}: {e}")
            return 0